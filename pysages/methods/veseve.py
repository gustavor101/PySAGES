# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Variational enhanced sampling in extended lagrangian using neural networks 
"""

from functools import partial

from jax import jit
from jax import grad
from jax import numpy as np
from jax import vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.grids import build_indexer
from pysages.methods.analysis import GradientLearning, _analyze
from pysages.methods.core import NNSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.ml.models import MLP
from pysages.ml.objectives import MeanLoss
from pysages.ml.optimizers import VESOptimizer
from pysages.ml.training import NNData, build_fitting_function, convolve, normalize
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver


class VESState(NamedTuple):
    """
    VES internal state.

    Parameters
    ----------

    xi: JaxArray (CV shape)
        CV of real variable (coords) recorded in the simulation.

    bias: JaxArray (natoms, 3)
        Array with biasing forces for each particle.

        nn: NNData
        Bundle of the neural network parameters, and output scaling coefficients.
        
    si: JaxArray (CV shape)
        CV in extended space
        
    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    si: JaxArray
    n_walkers: int #Remove if we use [n_walkers, cv_dim] otherwise keep if [cv_dim]
    gamma_s: float
    gamma_theta: float
    beta_hat: float
    bias: JaxArray
    nn: NNData
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialVESState(NamedTuple):
    xi: JaxArray
    si: JaxArray
    n_walkers: int
    gamma_s: float
    gamma_theta: float
    beta_hat: float
    ind: Tuple
    nn: NNData
    pred: bool


class VES(NNSamplingMethod):
    """
    Implementation of the sampling method described in
    ---

    Parameters
    ----------
    cvs: Union[List, Tuple]
        List of collective variables.

    topology: Tuple[int]
        Defines the architecture of the neural network
        (number of nodes of each hidden layer).

    train_freq: Optional[int] = 5000
        Training frequency.

    optimizer: Optional[Optimizer]
        Optimization method used for training, defaults to KL().

    restraints: Optional[CVRestraints] = None
        If provided, indicate that harmonic restraints will be applied when any
        collective variable lies outside the box from `restraints.lower` to
        `restraints.upper`.

    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, topology, **kwargs):
        super().__init__(cvs, topology, **kwargs)

        self.train_freq = kwargs.get("train_freq", 5000)

        # Neural network and optimizer intialization
        dims = xi.shape.size
        scale = partial(_scale, grid=grid)
        self.model = MLP(dims, dims, topology, transform=scale)
        default_optimizer = LevenbergMarquardt(reg=L2Regularization(1e-6))
        self.optimizer = kwargs.get("optimizer", default_optimizer)
        
    # How to estimate the energy
    def build(self, snapshot, helpers):
        return _ves(self, snapshot, helpers) #?


def _ves(method, snapshot, helpers):
    cv = method.cv
    train_freq = method.train_freq

    dt = snapshot.dt
    dims = xi.shape.size
    natoms = np.size(snapshot.positions, 0)

    # Neural network and optimizer
    ps, _ = unpack(method.model.parameters)

    # Helper methods
    learn_free_energy_grad = build_free_energy_grad_learner(method)
    estimate_free_energy_grad = build_force_estimator(method)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        cv_walkers = jnp.zeros([n_walker, cv_dim]) #?
        nn = NNData(ps, F, F)
        return VESState(xi, si, n_walkers, gamma_s, gamma_theta, bias, nn, 0)

    
    def update(state, data):
        ncalls = state.ncalls + 1
        in_training_regime = ncalls > 2 * train_freq
        in_training_step = in_training_regime & (ncalls % train_freq == 1)
        
        # NN training
        nn = learn_free_energy_grad(state, in_training_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)

        # Extended Space Walkers

        dz = grad(jnp.sum(student_model(zw)))
        dt = snapshot.dt
        weiner_si = jnp.sqrt(2*gamma_s)*jnp.array(np.random.randn(n_walker, cv_dim))
        
        si = si - dt * dz 
            + np.sqrt(2 * dt / beta_hat) * weiner_si

        F = estimate_free_energy_grad(
            PartialVESState(xi, si, n_walkers, gamma_s, gamma_theta, beta_hat, ind, nn, pred)
        )
        
        bias = (1-beta_hat)(Jxi.T @ F).reshape(state.bias.shape)
        #

        return VESState(xi, si, n_walkers, gamma_s, gamma_theta, bias, nn, ncalls)
    
    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_grad_learner(method: VES):
    """
    Returns a function that given a `VESState` trains the method's neural network
    parameters from an estimate of the free energy.

    The training data is regularized by convolving it with a Blackman window.
    """
    dims = xi.shape.size
    model = method.model

    # Training data
    inputs = xi

    _, layout = unpack(model.parameters)
    fit = build_fitting_function(model, method.optimizer)

    def train(nn, x, y):
        axes = tuple(range(y.ndim - 1))
        y, mean, std = normalize(y, axes=axes)
        reference = smooth(y)
        params = fit(nn.params, x, reference).params
        return NNData(params, mean, std / reference.std(axis=axes))

    def skip_learning(state):
        return state.nn

    def _learn_free_energy_grad(state, in_training_step):
        return cond(in_training_step, learn_free_energy_grad, skip_learning, state)

    return _learn_free_energy_grad


def build_force_estimator(method: VES):
    """
    Returns a function that given the neural network parameters and a CV value,
    evaluates the network on the provided CV.
    """
    f32 = np.float32
    f64 = np.float64

    N = method.N
    model = method.model
    grid = method.grid
    _, layout = unpack(model.parameters)

    def average_force(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(N, state.hist[i])

    def predict_force(state):
        nn = state.nn
        x = state.xi
        params = pack(nn.params, layout)
        return nn.std * f64(model.apply(params, f32(x)).flatten()) + nn.mean

    def _estimate_force(state):
        return cond(state.pred, predict_force, average_force, state)

    if method.restraints is None:
        estimate_force = _estimate_force
    else:
        lo, hi, kl, kh = method.restraints

        def restraints_force(state):
            xi = state.xi.reshape(grid.shape.size)
            return apply_restraints(lo, hi, kl, kh, xi)

        def estimate_force(state):
            ob = np.any(np.array(state.ind) == grid.shape)  # Out of bounds condition
            return cond(ob, restraints_force, _estimate_force, state)

    return estimate_force


@dispatch
def analyze(result: Result[VES], **kwargs):
    """
    Parameters
    ----------

    result: Result[VES]
        Result bundle containing the method, final states, and callbacks.

    topology: Optional[Tuple[int]] = result.method.topology
        Defines the architecture of the neural network
        (number of nodes in each hidden layer).

    Returns
    -------

    dict:
        A dictionary with the following keys:

        histogram: JaxArray
            A histogram of the visits to each bin in the CV grid.

        free_energy: JaxArray
            Free energy at each bin in the CV grid.

        mesh: JaxArray
            These are the values of the CVs that are used as inputs for training.

        nn: NNData
            Coefficients of the basis functions expansion approximating the free energy.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the
            CV domain defined by the grid.

    NOTE:
    For multiple-replicas runs we return a list (one item per-replica)
    for each attribute.
    """
    topology = kwargs.get("topology", result.method.topology)
    _result = _analyze(result, GradientLearning(), topology)
    _result["nn"] = first_or_all([state.nn for state in result.states])
    return numpyfy_vals(_result)
