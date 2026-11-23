# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Steered Spectral Adaptive Biasing Force Sampling Method.

SteeredSABF is Spectral ABF with external (e.g. funnel) restraints, plus a harmonic
restraint on the collective variables whose center moves at a prescribed velocity
(steered MD). The moving restraint drags the system back and forth along the collective
variables, so that the number of crossings between the end states is set by the protocol
instead of by diffusion, while the adaptive bias flattens the landscape underneath.

Bookkeeping (all per bin of the CV grid, `W` is the pseudo-inverse of the CV Jacobian):

- `Fsum` accumulates `d(W p)/dt` plus the ABF bias that was applied. The steering and the
  external restraints are *not* added back, so `Fsum / hist` is the mean force of the
  system including them.
- `Fsteer` accumulates the steering force that was applied. `(Fsum + Fsteer) / hist` is the
  mean force with the steering removed: this is the estimator to integrate when the
  external restraints are meant to be part of the Hamiltonian (e.g. a funnel, or a
  restraint that is only active in part of the CV range).
- `Frestr` accumulates the projection of the external restraint forces on the CVs.
  `(Fsum + Fsteer + Frestr) / hist` removes that projection as well.
- `hist_dir`, `Fsum_dir` and `Fsteer_dir` hold the same sums separately for the steps in
  which the center was moving forward (index 0) and backward (index 1); steps in which the
  center was held still only enter the totals. The difference between the two directions
  measures the hysteresis of the pulling protocol, and their equal-weight average (the
  `symmetric_*` results of `analyze`) cancels the part of the dissipative force that is
  odd in the pulling velocity even when the two directions were not sampled equally.
"""

from jax import jit
from jax import numpy as np
from jax.lax import cond

from pysages.approxfun import (
    Fun,
    SpectralGradientFit,
    build_evaluator,
    build_fitter,
    build_grad_evaluator,
    compute_mesh,
)
from pysages.colvars import get_periods, wrap
from pysages.grids import Chebyshev, Grid, build_indexer, convert, grid_transposer
from pysages.methods.core import GriddedSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver


class SteeredSABFState(NamedTuple):
    """
    SteeredSABF internal state.

    Parameters
    ----------

    xi: JaxArray [cvs.shape]
        Last collective variable recorded in the simulation.

    bias: JaxArray [natoms, 3]
        Array with biasing forces for each particle.

    hist: JaxArray [grid.shape]
        Histogram of visits to the bins in the collective variable grid.

    Fsum: JaxArray [grid.shape, cvs.shape]
        The cumulative force recorded at each bin of the CV grid (the steering and the
        external restraints are part of it).

    force: JaxArray [cvs.shape]
        Last estimate of the adaptive biasing force (the one applied in the last step).

    Wp: JaxArray [cvs.shape]
        Estimate of the product $W p$ where `p` is the matrix of momenta and
        `W` the Moore-Penrose inverse of the Jacobian of the CVs.

    Wp_: JaxArray [cvs.shape]
        The value of `Wp` for the previous integration step.

    fun: Fun
        Object that holds the coefficients of the basis functions
        approximation to the free energy.

    restr: JaxArray [cvs.shape]
        Last projection of the external restraint forces on the CVs.

    proj: JaxArray
        Last value logged by the external restraint.

    Frestr: JaxArray [grid.shape, cvs.shape]
        The cumulative projection of the external restraint forces at each bin of the grid.

    center: JaxArray [cvs.shape]
        Current center of the moving harmonic restraint.

    direction: float
        `+1.0` while the center moves along `velocity`, `-1.0` while it moves against it.

    dwell: int
        Number of steps the center will still be held at a turning point.

    sweeps: int
        Number of turning points reached so far (completed one-way sweeps).

    moved: float
        Motion of the center in the last step: `+1.0` forward, `-1.0` backward, `0.0`
        held still. It labels the forces that were applied in that step.

    kforce: JaxArray [cvs.shape]
        Last steering force (gradient of the harmonic restraint with respect to the CVs).

    Fsteer: JaxArray [grid.shape, cvs.shape]
        The cumulative steering force at each bin of the CV grid.

    hist_dir: JaxArray [2, grid.shape]
        `hist` restricted to the steps with the center moving forward (0) / backward (1).

    Fsum_dir: JaxArray [2, grid.shape, cvs.shape]
        `Fsum` restricted to the steps with the center moving forward (0) / backward (1).

    Fsteer_dir: JaxArray [2, grid.shape, cvs.shape]
        `Fsteer` restricted to the steps with the center moving forward (0) / backward (1).

    work: float
        Accumulated `-(force + kforce) . dc`, with `dc` the displacement of the center.

    ework: float
        Same as `work` with the projection of the external restraints included.

    work_steer: float
        Accumulated `-kforce . dc`: the work done by moving the harmonic restraint.

    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    force: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    fun: Fun
    restr: JaxArray
    proj: JaxArray
    Frestr: JaxArray
    center: JaxArray
    direction: float
    dwell: int
    sweeps: int
    moved: float
    kforce: JaxArray
    Fsteer: JaxArray
    hist_dir: JaxArray
    Fsum_dir: JaxArray
    Fsteer_dir: JaxArray
    work: float
    ework: float
    work_steer: float
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialSteeredSABFState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    Fsteer: JaxArray
    ind: Tuple
    fun: Fun
    pred: bool


class SteeredSABF(GriddedSamplingMethod):
    """
    Spectral ABF with external restraints and a moving harmonic restraint on the CVs.

    Parameters
    ----------

    cvs: Union[List, Tuple]
        Set of user selected collective variable.

    grid: Grid
        Specifies the collective variables domain and number of bins for discretizing
        the CV space along each CV dimension. For non-periodic grids this will be
        converted to a Chebyshev-distributed grid.

    kspring: Union[float, JaxArray]
        Force constant of the moving harmonic restraint: a scalar (same for every CV),
        one value per CV, or a full `(ncvs, ncvs)` matrix. Use zero for the CVs that
        should not be steered.

    center: Union[float, JaxArray]
        Initial center of the moving harmonic restraint (one value per CV).

    velocity: Union[float, JaxArray]
        Velocity of the center (one value per CV, CV units per unit of time).

    center_lower: Optional[Union[float, JaxArray]] = None
        Lower turning point of the center. Only used for CVs with nonzero velocity.

    center_upper: Optional[Union[float, JaxArray]] = None
        Upper turning point of the center. Only used for CVs with nonzero velocity.
        When neither bound is given the center moves at constant velocity for the whole
        run. Otherwise the center moves back and forth along a straight line in CV space:
        as soon as one steered CV reaches one of its turning points the direction of
        motion is reversed for all of them, so with several steered CVs the turning
        points of the others are only reached if they are reached at the same time.
        For periodic CVs the center and its turning points must lie within half a period
        of zero.

    dwell: Optional[int] = 0
        Number of steps the center is held still at each turning point.

    steer_in_bias: Optional[bool] = False
        What the adaptive bias is fitted to. If `False` (default) it is fitted to
        `(Fsum + Fsteer) / hist`, the mean force of the unsteered system, so that the
        bias converges to the free energy and the moving restraint only has to drag the
        system over a flat landscape. If `True` it is fitted to `Fsum / hist`, that is,
        the steering force is treated as part of the Hamiltonian and its bin average is
        compensated by the bias; the load is then shared arbitrarily between the bias
        and the restraint, which can settle into large opposing forces. The accumulated
        quantities and the results of `analyze` are defined in the same way in both cases.

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average
        of the adaptive biasing force.

    fit_freq: Optional[int] = 100
        Fitting frequency.

    fit_threshold: Optional[int] = 500
        Number of time steps after which fitting starts to take place.

    restraints: Optional[CVRestraints] = None
        If provided, indicate that harmonic restraints will be applied when any
        collective variable lies outside the box from `restraints.lower` to
        `restraints.upper`.

    use_pinv: Optional[Bool] = False
        If set to True, the product `W @ p` will be estimated using
        `np.linalg.pinv` rather than using the `scipy.linalg.solve` function.
        This is computationally more expensive but numerically more stable.

    ext_force: Optional[Callable] = None
        External restraints. A function that takes the snapshot data and returns the
        gradient of the restraint energy with respect to the positions, and a value to
        log (same contract as `get_funnel_force`).
    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, grid, kspring, center, velocity, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.N = np.asarray(self.kwargs.get("N", 500))
        self.fit_freq = self.kwargs.get("fit_freq", 100)
        self.fit_threshold = self.kwargs.get("fit_threshold", 500)
        self.grid = self.grid if self.grid.is_periodic else convert(self.grid, Grid[Chebyshev])
        self.model = SpectralGradientFit(self.grid)
        self.use_pinv = self.kwargs.get("use_pinv", False)
        self.steer_in_bias = bool(self.kwargs.get("steer_in_bias", False))

        d = len(self.cvs)
        self.kspring = _spring_matrix(kspring, d)
        self.center = _per_cv("center", center, d)
        self.velocity = _per_cv("velocity", velocity, d)
        lower = self.kwargs.get("center_lower", None)
        upper = self.kwargs.get("center_upper", None)
        steered = self.velocity != 0
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        self.center_lower = np.where(steered, _per_cv("center_lower", lower, d), -np.inf)
        self.center_upper = np.where(steered, _per_cv("center_upper", upper, d), np.inf)
        self.dwell = int(self.kwargs.get("dwell", 0))

        if np.any(self.center_lower >= self.center_upper):
            raise ValueError("`center_lower` must be smaller than `center_upper` for every CV.")
        if np.any((self.center < self.center_lower) | (self.center > self.center_upper)):
            raise ValueError("`center` must lie between `center_lower` and `center_upper`.")
        if self.dwell < 0:
            raise ValueError("`dwell` must be a non-negative number of steps.")
        half_period = get_periods(self.cvs) / 2
        for value in (self.center, self.center_lower, self.center_upper):
            if np.any(np.isfinite(value) & (np.abs(value) > half_period)):
                raise ValueError(
                    "For periodic CVs, `center` and its turning points must lie within half "
                    "a period of zero."
                )

    def build(self, snapshot, helpers, *_args, **_kwargs):
        """
        Returns the `initialize` and `update` functions for the sampling method.
        """
        self.ext_force = self.kwargs.get("ext_force", None)
        return _steered_sabf(self, snapshot, helpers)


def _per_cv(name, value, d):
    value = np.asarray(value, dtype=float).flatten()
    value = np.full(d, value[0]) if value.size == 1 else value
    if value.size != d:
        raise ValueError(f"`{name}` must be a scalar or have one value per CV ({d}).")
    return value


def _spring_matrix(kspring, d):
    k = np.asarray(kspring, dtype=float)
    if k.ndim == 2 and k.shape != (d, d):
        raise ValueError(f"A `kspring` matrix must have shape ({d}, {d}).")
    if k.ndim == 2 and not np.allclose(k, k.T):
        raise ValueError("A `kspring` matrix must be symmetric.")
    return k if k.ndim == 2 else np.diag(_per_cv("kspring", k, d))


def _steered_sabf(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    fit_freq = method.fit_freq
    fit_threshold = method.fit_threshold

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)

    kspring = method.kspring
    velocity = method.velocity
    c_lo = method.center_lower
    c_hi = method.center_upper
    dwell_steps = method.dwell
    periods = get_periods(method.cvs)

    # Helper methods
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    fit = build_fitter(method.model)
    fit_forces = build_free_energy_fitter(method, fit)
    estimate_force = build_force_estimator(method)

    if method.ext_force is None:

        def ext_force(data):
            return np.zeros((natoms, helpers.dimensionality())), 0.0

    else:
        ext_force = method.ext_force

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(Fsum)
        restr = np.zeros(dims)
        proj = 0.0
        Frestr = np.zeros((*grid.shape, dims))
        kforce = np.zeros(dims)
        Fsteer = np.zeros((*grid.shape, dims))
        hist_dir = np.zeros((2, *grid.shape), dtype=np.uint32)
        Fsum_dir = np.zeros((2, *grid.shape, dims))
        Fsteer_dir = np.zeros((2, *grid.shape, dims))
        return SteeredSABFState(
            xi,
            bias,
            hist,
            Fsum,
            force,
            Wp,
            Wp_,
            fun,
            restr,
            proj,
            Frestr,
            method.center,
            1.0,
            0,
            0,
            0.0,
            kforce,
            Fsteer,
            hist_dir,
            Fsum_dir,
            Fsteer_dir,
            0.0,
            0.0,
            0.0,
            0,
        )

    def update(state, data):
        # During the intial stage use ABF
        ncalls = state.ncalls + 1
        in_fitting_regime = ncalls > fit_threshold
        in_fitting_step = in_fitting_regime & (ncalls % fit_freq == 1)
        # Fit forces
        fun = fit_forces(state, in_fitting_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        # Restraint force and logger
        e_f, proj = ext_force(data)
        #
        p = data.momenta
        Wp = tsolve(Jxi, p)
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        # The sample belongs to the forces applied in the last step: only the adaptive
        # bias is added back (the steering and the external restraints stay in `Fsum`),
        # and the steering and restraint forces that were applied are accumulated on
        # their own. The first two samples of a run are discarded, since the finite
        # difference has no momentum history yet.
        I_xi = get_grid_index(xi)
        valid = np.asarray(ncalls > 2)
        v = np.where(valid, 1.0, 0.0)
        sample = dWp_dt + state.force
        hist = state.hist.at[I_xi].add(valid.astype(state.hist.dtype))
        Fsum = state.Fsum.at[I_xi].add(v * sample)
        Fsteer = state.Fsteer.at[I_xi].add(v * state.kforce)
        Frestr = state.Frestr.at[I_xi].add(v * state.restr)
        # Same sums, separately for the forward and backward motion of the center
        moved = valid & (state.moved != 0)
        I_dir = (np.where(state.moved > 0, 0, 1), *I_xi)
        w = np.where(moved, 1.0, 0.0)
        hist_dir = state.hist_dir.at[I_dir].add(moved.astype(state.hist_dir.dtype))
        Fsum_dir = state.Fsum_dir.at[I_dir].add(w * sample)
        Fsteer_dir = state.Fsteer_dir.at[I_dir].add(w * state.kforce)
        #
        force = estimate_force(
            PartialSteeredSABFState(xi, hist, Fsum, Fsteer, I_xi, fun, in_fitting_regime)
        )
        # Steering force (gradient of the harmonic restraint with respect to the CVs)
        kforce = kspring @ wrap(xi.reshape(dims) - state.center, periods)
        bias = np.reshape(-Jxi.T @ (force + kforce), state.bias.shape) + np.reshape(
            -e_f, state.bias.shape
        )
        # Restraint contribution to force
        restr = tsolve(Jxi, e_f.reshape(p.shape))
        # Move the center (it is held still while dwelling at a turning point). A step
        # that would cross a turning point is shortened so that the center lands on it.
        moving = np.asarray(state.dwell == 0)
        step = np.where(moving, state.direction, 0.0) * velocity * dt
        room = np.where(step > 0, c_hi - state.center, c_lo - state.center)
        reach = np.min(np.where(step != 0, room / np.where(step != 0, step, 1.0), np.inf))
        hit = reach <= 1.0
        dc = np.clip(reach, 0.0, 1.0) * step
        work = state.work - (force + kforce) @ dc
        ework = state.ework - (force + kforce + restr) @ dc
        work_steer = state.work_steer - kforce @ dc
        center = wrap(np.clip(state.center + dc, c_lo, c_hi), periods)
        direction = np.where(hit, -state.direction, state.direction)
        dwell = np.where(hit, dwell_steps, np.maximum(state.dwell - 1, 0))
        sweeps = state.sweeps + hit
        return SteeredSABFState(
            xi,
            bias,
            hist,
            Fsum,
            force,
            Wp,
            state.Wp,
            fun,
            restr,
            proj,
            Frestr,
            center,
            direction,
            dwell,
            sweeps,
            np.where(moving, state.direction, 0.0),
            kforce,
            Fsteer,
            hist_dir,
            Fsum_dir,
            Fsteer_dir,
            work,
            ework,
            work_steer,
            ncalls,
        )

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_fitter(method: SteeredSABF, fit):
    """
    Returns a function that given a `SteeredSABFState` performs a least squares fit of the
    generalized average forces for finding the coefficients of a basis functions expansion
    of the free energy.
    """
    # Weight of the accumulated steering force in the forces the bias is fitted to
    a = 0.0 if method.steer_in_bias else 1.0

    def _fit_forces(state):
        shape = (*state.Fsum.shape[:-1], 1)
        force = (state.Fsum + a * state.Fsteer) / np.maximum(state.hist.reshape(shape), 1)
        return fit(force)

    def skip_fitting(state):
        return state.fun

    def fit_forces(state, in_fitting_step):
        return cond(in_fitting_step, _fit_forces, skip_fitting, state)

    return fit_forces


@dispatch
def build_force_estimator(method: SteeredSABF):
    """
    Returns a function that given the coefficients of basis functions expansion and a CV
    value, evaluates the function approximation to the gradient of the free energy.
    """
    N = method.N
    grid = method.grid
    dims = grid.shape.size
    model = method.model
    get_grad = build_grad_evaluator(model)
    a = 0.0 if method.steer_in_bias else 1.0

    def average_force(state):
        i = state.ind
        return (state.Fsum[i] + a * state.Fsteer[i]) / np.maximum(state.hist[i], N)

    def interpolate_force(state):
        return get_grad(state.fun, state.xi).reshape(grid.shape.size)

    def _estimate_force(state):
        return cond(state.pred, interpolate_force, average_force, state)

    if method.restraints is None:
        ob_force = jit(lambda state: np.zeros(dims))
    else:
        lo, hi, kl, kh = method.restraints

        def ob_force(state):
            xi = state.xi.reshape(grid.shape.size)
            return apply_restraints(lo, hi, kl, kh, xi)

    def estimate_force(state):
        ob = np.any(np.array(state.ind) == grid.shape)  # Out of bounds condition
        return cond(ob, ob_force, _estimate_force, state)

    return estimate_force


@dispatch
def analyze(result: Result[SteeredSABF]):
    """
    Parameters
    ----------

    result: Result[SteeredSABF]
        Result bundle containing the method, final states, and callbacks.

    dict:
        A dictionary with the following keys:

        histogram: JaxArray
            A histogram of the visits to each bin in the CV grid.

        mean_force: JaxArray
            Generalized mean forces at each bin in the CV grid, as accumulated (`Fsum`):
            the steering and the external restraints are part of them.

        free_energy: JaxArray
            Free energy at each bin in the CV grid from the expansion that biases the
            simulation.

        mesh: JaxArray
            These are the values of the CVs that are used as inputs for training.

        fun: Fun
            Coefficients of the basis functions expansion that biases the simulation.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate `free_energy` in the CV domain defined
            by the grid.

        steer_force: JaxArray
            Average steering force at each bin in the CV grid.

        unsteered_force: JaxArray
            Generalized mean forces with the steering removed, `(Fsum + Fsteer) / hist`.
            The external restraints are still part of them.

        unsteered_energy: JaxArray
            Free energy fitted to `unsteered_force` (external restraints included).

        fun_unsteered: Fun
            Coefficients of the basis functions expansion for `unsteered_energy`.

        unsteered_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate `unsteered_energy`.

        unsteered_force_forward, unsteered_force_backward: JaxArray
            `unsteered_force` from the steps with the center moving forward / backward.

        unsteered_energy_forward, unsteered_energy_backward: JaxArray
            Free energies fitted to the forward / backward forces. Their difference
            measures the hysteresis of the pulling protocol.

        histogram_forward, histogram_backward: JaxArray
            Histograms of the steps with the center moving forward / backward.

        symmetric_force: JaxArray
            `unsteered_force` with the forward and backward mean forces averaged with
            equal weights in every bin visited in both directions (the steps with the
            center held still keep their own weight; bins visited in only one direction
            fall back to `unsteered_force`).

        symmetric_energy: JaxArray
            Free energy fitted to `symmetric_force` (external restraints included).

        fun_symmetric: Fun
            Coefficients of the basis functions expansion for `symmetric_energy`.

        symmetric_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate `symmetric_energy`.

        corrected_force: JaxArray
            Generalized mean forces without steering and without the projection of the
            external restraints at each bin in the CV grid.

        corrected_energy: JaxArray
            Free energy fitted to `corrected_force`.

        fun_corr: Fun
            Coefficients of the basis functions expansion for `corrected_energy`.

        corr_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate `corrected_energy`.

        restraint_force: JaxArray
            Generalized mean forces of the restraints at each bin in the CV grid.

        restraint_energy: JaxArray
            Free energy of restraints at each bin in the CV grid.

        fun_rstr: Fun
            Coefficients of the basis functions expansion for `restraint_energy`.

        rstr_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate `restraint_energy`.

        center, direction, sweeps: JaxArray
            Final center of the moving restraint, its direction of motion, and the
            number of turning points reached.

        work, ework, work_steer: JaxArray
            Accumulated work (see `SteeredSABFState`).

    NOTE:
    For multiple-replicas runs we return a list (one item per-replica) for each attribute.

    NOTE:
    The `*_energy` results are least squares fits over the whole grid, in which the bins
    that were never visited enter with zero mean force. When only part of the grid is
    sampled, integrate the binned `*_force` results over the visited bins instead.
    """
    method = result.method
    fit = build_fitter(method.model)
    grid = method.grid
    mesh = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    evaluate = build_evaluator(method.model)

    def average_forces(hist, Fsum):
        hist = np.expand_dims(hist, hist.ndim)
        return Fsum / np.maximum(hist, 1)

    def fit_average(hist, Fsum):
        return fit(average_forces(hist, Fsum))

    def build_fes_fn(fun):
        def fes_fn(x):
            A = evaluate(fun, x)
            return A.max() - A

        return jit(fes_fn)

    def symmetric_forces(s):
        # Equal weights for the two directions of motion wherever both were sampled
        n_f, n_b = (np.expand_dims(1.0 * h, h.ndim) for h in s.hist_dir)
        n = np.expand_dims(1.0 * s.hist, s.hist.ndim)
        F_f, F_b = s.Fsum_dir + s.Fsteer_dir
        F = s.Fsum + s.Fsteer
        F_held = F - F_f - F_b
        mean = 0.5 * (F_f / np.maximum(n_f, 1) + F_b / np.maximum(n_b, 1))
        balanced = ((n_f + n_b) * mean + F_held) / np.maximum(n, 1)
        return np.where((n_f > 0) & (n_b > 0), balanced, F / np.maximum(n, 1))

    keys = (
        "histogram",
        "mean_force",
        "free_energy",
        "fun",
        "fes_fn",
        "steer_force",
        "unsteered_force",
        "unsteered_energy",
        "fun_unsteered",
        "unsteered_fn",
        "histogram_forward",
        "histogram_backward",
        "unsteered_force_forward",
        "unsteered_force_backward",
        "unsteered_energy_forward",
        "unsteered_energy_backward",
        "symmetric_force",
        "symmetric_energy",
        "fun_symmetric",
        "symmetric_fn",
        "corrected_force",
        "corrected_energy",
        "fun_corr",
        "corr_fn",
        "restraint_force",
        "restraint_energy",
        "fun_rstr",
        "rstr_fn",
        "center",
        "direction",
        "sweeps",
        "work",
        "ework",
        "work_steer",
    )
    out = {key: [] for key in keys}
    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for s in result.states:
        F_uns = s.Fsum + s.Fsteer
        F_corr = F_uns + s.Frestr
        fun_uns = fit_average(s.hist, F_uns)
        fun_corr = fit_average(s.hist, F_corr)
        fun_rstr = fit_average(s.hist, s.Frestr)
        fes_fn = build_fes_fn(s.fun)
        uns_fn = build_fes_fn(fun_uns)
        corr_fn = build_fes_fn(fun_corr)
        rstr_fn = build_fes_fn(fun_rstr)

        out["histogram"].append(transpose(s.hist))
        out["mean_force"].append(transpose(average_forces(s.hist, s.Fsum)))
        out["free_energy"].append(transpose(fes_fn(mesh)))
        out["fun"].append(s.fun)
        out["fes_fn"].append(fes_fn)
        out["steer_force"].append(transpose(average_forces(s.hist, s.Fsteer)))
        out["unsteered_force"].append(transpose(average_forces(s.hist, F_uns)))
        out["unsteered_energy"].append(transpose(uns_fn(mesh)))
        out["fun_unsteered"].append(fun_uns)
        out["unsteered_fn"].append(uns_fn)
        for k, tag in enumerate(("forward", "backward")):
            F_dir = s.Fsum_dir[k] + s.Fsteer_dir[k]
            dir_fn = build_fes_fn(fit_average(s.hist_dir[k], F_dir))
            out["histogram_" + tag].append(transpose(s.hist_dir[k]))
            out["unsteered_force_" + tag].append(transpose(average_forces(s.hist_dir[k], F_dir)))
            out["unsteered_energy_" + tag].append(transpose(dir_fn(mesh)))
        F_sym = symmetric_forces(s)
        fun_sym = fit(F_sym)
        sym_fn = build_fes_fn(fun_sym)
        out["symmetric_force"].append(transpose(F_sym))
        out["symmetric_energy"].append(transpose(sym_fn(mesh)))
        out["fun_symmetric"].append(fun_sym)
        out["symmetric_fn"].append(sym_fn)
        out["corrected_force"].append(transpose(average_forces(s.hist, F_corr)))
        out["corrected_energy"].append(transpose(corr_fn(mesh)))
        out["fun_corr"].append(fun_corr)
        out["corr_fn"].append(corr_fn)
        out["restraint_force"].append(transpose(average_forces(s.hist, s.Frestr)))
        out["restraint_energy"].append(transpose(rstr_fn(mesh)))
        out["fun_rstr"].append(fun_rstr)
        out["rstr_fn"].append(rstr_fn)
        for key in ("center", "direction", "sweeps", "work", "ework", "work_steer"):
            out[key].append(np.asarray(getattr(s, key)))

    ana_result = {key: first_or_all(val) for key, val in out.items()}
    ana_result["mesh"] = transpose(mesh).reshape(-1, d).squeeze()

    return numpyfy_vals(ana_result)
