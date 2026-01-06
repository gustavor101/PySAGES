# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Sobolev KDE (SKDE).

SKDE learns the generalized mean forces and frequencies as a function of some collective variables,
by training a neural network from a kde of the mean forces and densities estimates.
It closely follows the Sirens method, except for using grids in
the simulation, which is done instead with a binless approximation to the
generalized mean force provided by the network.

"""

from functools import partial

from jax import grad, jit
from jax import numpy as np
from jax import random, value_and_grad, vmap
from jax.lax import cond
from objectives import L2Regularization, Sobolev1SSE
from optimizers import JaxOptimizer

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.colvars import wrap
from pysages.grids import build_indexer, grid_transposer
from pysages.methods.core import NNSamplingMethod, Result, generalize
from pysages.methods.metad import sum_of_gaussians
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.ml.models import Siren
from pysages.ml.training import NNData, build_fitting_function
from pysages.ml.utils import pack, unpack
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver, try_import

jopt = try_import("jax.example_libraries.optimizers", "jax.experimental.optimizers")


class SKDEState(NamedTuple):
    """
    SKDE internal state.

    Parameters
    ----------

    xi: JaxArray (CV shape)
        Last collective variable recorded in the simulation.

    bias: JaxArray (natoms, 3)
        Array with biasing forces for each particle.

    hist: JaxArray (grid.shape)
        Histogram of visits to the bins in the collective variable grid.

    Fsum: JaxArray (grid.shape, CV shape)
        The cumulative force recorded at each bin of the CV grid.

    Wp: JaxArray (CV shape)
        Estimate of the product $W p$ where `p` is the matrix of momenta and
        `W` the Moore-Penrose inverse of the Jacobian of the CVs.

    Wp_: JaxArray (CV shape)
        The value of `Wp` for the previous integration step.

    nn: NNData
        Bundle of the neural network parameters, and output scaling coefficients.

    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    F: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    phi: JaxArray
    nn: NNData
    si: JaxArray
    key: JaxArray
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialSKDEState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    nn: NNData
    pred: bool


class SKDE(NNSamplingMethod):
    """
    Implementation of the sampling method described in
    "On the Fly Multidimensional Enhanced Sampling with Functions in Sobolev Spaces"
    .

    Parameters
    ----------
    cvs: Union[List, Tuple]
        List of collective variables.

    grid: Grid
        Specifies the CV domain and number of bins for discretizing the CV space
        along each CV dimension.

    topology: Tuple[int]
        Defines the architecture of the neural network
        (number of nodes of each hidden layer).

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average of the
        binned generalized mean force.

    train_freq: Optional[int] = 5000
        Training frequency.

    optimizer: Optional[Optimizer]
        Optimization method used for training, defaults to LevenbergMarquardt().

    restraints: Optional[CVRestraints] = None
        If provided, indicate that harmonic restraints will be applied when any
        collective variable lies outside the box from `restraints.lower` to
        `restraints.upper`.

    use_pinv: Optional[Bool] = False
        If set to True, the product `W @ p` will be estimated using
        `np.linalg.pinv` rather than using the `scipy.linalg.solve` function.
        This is computationally more expensive but numerically more stable.
    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, grid, topology, **kwargs):
        super().__init__(cvs, grid, topology, **kwargs)
        self.nbatches = np.asarray(kwargs.get("nbatches", 5000))
        self.N = np.asarray(kwargs.get("N", 500))
        self.kT = np.asarray(kwargs.get("kT", 1.0))
        self.rho = np.asarray(kwargs.get("rho", 0.6))
        self.train_freq = kwargs.get("train_freq", 5000)
        self.biasfactor = kwargs.get("biasfactor", 20)
        self.ntrains = kwargs.get("ntrains", 10000)
        self.si = kwargs.get("si", 20000)
        self.sigma = kwargs.get("sigma", 0.241827118)
        self.periods = kwargs.get("periods", 0.0)

        # Neural network and optimizer intialization
        dims = grid.shape.size
        scale = partial(_scale, grid=grid)
        self.model = Siren(dims, 1, topology, transform=scale)
        loss = Sobolev1SSE()
        default_optimizer = JaxOptimizer(jopt.adamw, loss=loss, tol=1e-6, reg=L2Regularization(0.0))
        self.optimizer = kwargs.get("optimizer", default_optimizer)
        self.use_pinv = self.kwargs.get("use_pinv", False)

    def build(self, snapshot, helpers):
        return _skde(self, snapshot, helpers)


def _skde(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq
    nbatches = method.nbatches
    N = method.N
    dt = snapshot.dt
    si = method.si
    gamma = np.where(method.biasfactor > 1.0, 1.0 - 1.0 / (method.biasfactor), 1.0)
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)

    # Neural network and optimizer
    ps, _ = unpack(method.model.parameters)

    # Helper methods
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    learn_free_energy_grad = build_free_energy_grad_learner(method)
    estimate_free_energy_grad = build_force_estimator(method)
    estimate_free_energy = build_energy_estimator(method)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        hist = np.zeros((nbatches, dims))
        Fsum = np.zeros((nbatches, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        phi = np.zeros(nbatches)
        nn = NNData(ps, np.array(0.0), np.array(1.0))
        key = random.PRNGKey(0)
        weiner_si = random.uniform(
            key, shape=(si, dims), minval=grid.lower.flatten(), maxval=grid.upper.flatten()
        )
        return SKDEState(xi, bias, hist, Fsum, F, Wp, Wp_, phi, nn, weiner_si, key, 0)

    def update(state, data):
        # During the intial stage, collect samples
        ncalls = state.ncalls + 1
        in_training_regime = ncalls > 1 * train_freq
        ntrains = ncalls / train_freq
        in_training_step = in_training_regime & (ncalls % train_freq == 1)
        # NN training, the key and new extended points for FES integration
        nn, si, key = learn_free_energy_grad(state, in_training_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        Wp = tsolve(Jxi, p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        # The grid index is used to apply restraints in CV space
        I_xi = get_grid_index(xi)
        # The index in the batch for training
        idx = state.ncalls % nbatches
        hist = state.hist.at[idx].set(xi.flatten())
        Fsum = state.Fsum.at[idx].set(dWp_dt.flatten() + state.F.flatten())
        # The force is scaled by the gamma factor in case of WT target
        F = estimate_free_energy_grad(
            PartialSKDEState(xi, hist, Fsum, I_xi, nn, in_training_regime)
        )
        force = np.where(ntrains > N, gamma * F, gamma * F * ntrains / N)
        fe = estimate_free_energy(PartialSKDEState(xi, hist, Fsum, I_xi, nn, in_training_regime))
        ener = np.where(ntrains > N, gamma * fe.sum(), gamma * fe.sum() * ntrains / N)
        phi = state.phi.at[idx].set(ener)
        bias = (-Jxi.T @ force).reshape(state.bias.shape)
        #
        return SKDEState(xi, bias, hist, Fsum, force, Wp, state.Wp, phi, nn, si, key, ncalls)

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_grad_learner(method: SKDE):
    """
    Returns a function that given a `SKDEState` trains the method's neural network
    parameters from a Sobolev estimate for the free energy.

    """

    grid = method.grid
    dims = grid.shape.size
    model = method.model
    kT = method.kT
    si_size = method.si
    rho = method.rho
    ntrains = method.ntrains
    train_freq = method.train_freq
    sigma = method.sigma
    P = method.periods
    gamma = np.where(method.biasfactor > 1.0, 1.0 - 1.0 / (method.biasfactor), 1.0)
    apply = value_and_grad(lambda p, x: model.apply(p, x.reshape(1, -1)).sum(), argnums=1)
    _, layout = unpack(model.parameters)
    fit = build_fitting_function(model, method.optimizer)

    def latin_hypercube_samples(rng_key, n_samples, n_dim, gridmin, gridmax):
        """
        Generates a basic Latin Hypercube Sample (LHS) in a grid defined by min and max values.

        Args:
            rng_key: JAX PRNG key.
            n_samples (int): The number of samples.
            n_dim (int): The number of dimensions.
            gridmin (Jax array): Min value of the grid
            gridmax (Jax array): Max value of the grid

        Returns:
            jax.Array: An array of shape (n_samples, n_dim) with LHS points.
        """
        # 1. Create a base grid of stratified intervals
        # Intervals are [0, 1/n_samples), [1/n_samples, 2/n_samples), ...
        intervals, step = np.linspace(0.0, 1.0, n_samples, endpoint=False, retstep=True)

        # 2. Sample a random point within each interval for each dimension
        # Generate random offsets in [0, 1) for each point and dimension
        key, subkey = random.split(rng_key)
        offsets = random.uniform(subkey, (n_samples, n_dim))

        # Scale offsets to the interval width (1/n_samples) and add to the start of intervals
        # This gives points centered within their respective strata in the dimension-wise sense.
        points = intervals.reshape((offsets.shape[0], 1)) + offsets * step

        # 3. Permute the samples along each dimension independently to ensure
        # only one sample per "row" and "column" in all 2D projections (for strength 1 LHS)
        for i in range(n_dim):
            key, subkey = random.split(key)
            points = random.permutation(subkey, points, axis=0, independent=True)
        points = points * (gridmax - gridmin) + gridmin
        return points, key

    def row_sum(x):
        """
        Sum array `x` along each of its row (`axis = 1`),
        """
        return np.sum(x.reshape(np.size(x, 0), -1), axis=1)

    def gaussianf(a, sigma, x):
        """
        N-dimensional origin-centered gaussian with height `a` and standard deviation `sigma`.
        """
        exponents = -row_sum((x / sigma) ** 2) / 2  # shape: (n,)
        # a: (n, m)
        # We want to broadcast exponents (n,) to (n, m)
        return a * np.exp(exponents)[:, None]

    def sum_of_gaussiansf(xi, heights, centers, sigmas, periods):
        """
        Sum of n-dimensional Gaussians potential for force estimation.
        """
        delta_x = wrap(xi - centers, periods)
        return gaussianf(heights, sigmas, delta_x).sum(axis=0)

    def train(nn, x, y):
        e, f, mean, std = y
        e = e - mean
        f = f / std
        e = e / std
        params = fit(nn.params, x, (e, f)).params
        return NNData(params, mean, std)

    def learn_free_energy_grad(state):
        hist = state.hist
        F = state.Fsum
        # Unbias the density estimation by score function (arxiv)
        params = pack(state.nn.params, layout)
        _, gradn = vmap(lambda x: apply(params, x))(hist)
        hist = hist + sigma * sigma * state.nn.std * gradn / 2.0
        hist = wrap(hist, P)
        k = np.floor(state.ncalls / train_freq)
        # Combine the LHS and the unbiased batch sampling for FES integration
        si = np.concatenate([state.si, hist], axis=0)
        fe0, grad0 = vmap(lambda x: apply(params, x))(si)
        femin = state.nn.std * fe0.min()
        fe0 = state.nn.std * (fe0 - fe0.min())
        grad0 = state.nn.std * grad0
        # Set the minimum in FES as in ANN method
        fe = state.phi - gamma * femin
        weights = np.exp(fe / kT)
        # KDE estimation of the frequencies
        sigma_n = sigma * sigma
        rho_p = vmap(
            lambda x: sum_of_gaussians(
                x, weights / (np.power(np.sqrt(2 * np.pi * sigma_n), dims)), state.hist, sigma, P
            )
        )(si)
        density = np.exp(fe0 / kT) + rho_p
        ener = kT * np.log(density)
        ener = ener.reshape((grad0.shape[0], 1))
        mean0 = ener.mean()
        # KDE estimation of the mean forces
        rho_f = vmap(
            lambda x: sum_of_gaussiansf(
                x, F / (np.power(np.sqrt(2 * np.pi * sigma_n), dims)), state.hist, sigma, P
            )
        )(si)
        rho_pf = vmap(
            lambda x: sum_of_gaussians(
                x, 1.0 / (np.power(np.sqrt(2 * np.pi * sigma_n), dims)), state.hist, sigma, P
            )
        )(si)
        probf = 1.0 / (1.0 + rho_pf)
        force = (grad0 + rho_f) * probf.reshape((grad0.shape[0], 1))
        # Training
        std0 = np.maximum(ener.std(), force.std(axis=0).max())
        nn_temp = train(state.nn, si, (ener, force, mean0, std0))
        params_now, _ = unpack(nn_temp.params)
        params_old, _ = unpack(state.nn.params)
        params_final = np.where(
            k < ntrains, params_now, rho * params_old + (1.0 - rho) * params_now
        )
        nn = NNData(params_final, mean0, std0)
        # New LHS points
        new_si, key = latin_hypercube_samples(
            state.key, si_size, dims, grid.lower.flatten(), grid.upper.flatten()
        )
        return nn, new_si, key

    def skip_learning(state):
        return state.nn, state.si, state.key

    def _learn_free_energy_grad(state, in_training_step):
        return cond(in_training_step, learn_free_energy_grad, skip_learning, state)

    return _learn_free_energy_grad


def build_force_estimator(method: SKDE):
    """
    Returns a function that given the neural network parameters and a CV value,
    evaluates the network on the provided CV.
    """
    f64 = np.float64

    model = method.model
    grid = method.grid
    dims = grid.shape.size
    _, layout = unpack(model.parameters)
    model_grad = grad(lambda p, x: model.apply(p, x).sum(), argnums=1)

    def average_force(state):

        return np.zeros(dims)

    def predict_force(state):
        nn = state.nn
        x = state.xi
        params = pack(nn.params, layout)
        return nn.std * f64(model_grad(params, x).flatten())

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


def build_energy_estimator(method: SKDE):
    """
    Returns a function that given the neural network parameters and a CV value,
    evaluates the network on the provided CV.
    """
    f64 = np.float64

    model = method.model
    _, layout = unpack(model.parameters)

    def average_energy(state):

        return np.zeros(1)

    def predict_energy(state):
        nn = state.nn
        x = state.xi
        params = pack(nn.params, layout)
        return (nn.std * f64(model.apply(params, x))).flatten()

    def estimate_energy(state):
        return cond(state.pred, predict_energy, average_energy, state)

    return estimate_energy


@dispatch
def analyze(result: Result[SKDE], **kwargs):
    """
    Computes the free energy from the result of an Sobolev-KDE run.

    Parameters
    ----------

    result: Result:
        Result bundle containing method, final Sirens-like state, and callback.

    strategy: SobolevLearning

    topology: Tuple[int]
        Defines the architecture of the neural network
        (number of nodes in each hidden layer).

    Returns
    -------

    dict: A dictionary with the following keys:

        histogram: JaxArray
            Histogram for the states visited during the method.

        mean_force: JaxArray
            Average force at each bin of the CV grid.

        free_energy: JaxArray
            Free Energy at each bin of the CV grid.

        mesh: JaxArray
            Grid used in the method.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the
            CV domain defined by the grid.

    NOTE:
    For multiple-replicas runs we return a list (one item per-replica)
    for each attribute.
    """

    method = result.method
    states = result.states
    oldmodel = method.model
    grid = method.grid
    mesh = inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    apply = grad(lambda p, x: oldmodel.apply(p, x.reshape(1, -1)).sum(), argnums=1)

    def build_fes_fn(state):
        _, layout = unpack(oldmodel.parameters)

        def fes_fn(x):
            params = pack(state.nn.params, layout)
            A = state.nn.std * oldmodel.apply(params, x) + state.nn.mean
            return A.max() - A

        return jit(fes_fn)

    def average_forces(state):
        _, oldlayout = unpack(oldmodel.parameters)
        oldparams = pack(state.nn.params, oldlayout)
        F = state.nn.std * vmap(lambda x: apply(oldparams, x))(inputs)
        F = F.reshape((*grid.shape, grid.shape.size))
        return F

    mean_forces = []
    free_energies = []
    fes_fns = []

    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for state in states:
        fes_fn = build_fes_fn(state)
        mean_forces.append(transpose(average_forces(state)))
        free_energies.append(transpose(fes_fn(mesh)))
        fes_fns.append(fes_fn)

    ana_result = {
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "fes_fn": first_or_all(fes_fns),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
    }
    return numpyfy_vals(ana_result)
