# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Online-Deposition Batch-Updated Variational Kernel Enhanced Sampling Methods.

Implements four walker-free, grid-free enhanced sampling methods on a bounded
local-linear Hermite Nadaraya-Watson Gaussian kernel basis with smooth support
gating and self-consistent partition-function normalization:

1. `DBKMetaD` (`DepositedBatchKernelMetaD`):
   Online-deposition logarithmic batch potential updates with energy-only
   weighted least-squares slope reconstruction.
2. `FOKMetaD` (`FirstOrderKernelMetaD`):
   First-order linear Maclaurin expansion of the batch potential update within
   its analytic trust region (`u <= u_max`), paired with energy-only slopes.
3. `DBKABF` (`DepositedBatchKernelABF`):
   Online-deposition generalized-momentum adaptive biasing force accumulation
   paired with batch-updated scalar potential offsets.
4. `DBCZAR` (`DepositedBatchCZAR`):
   Momentum-free online Well-Tempered Corrected z-Averaged Restraint (CZAR)
   batch score updates paired with scalar potential offsets.
"""

from jax import grad, jit, value_and_grad
from jax import numpy as np
from jax.lax import cond, map as lax_map

from pysages.approxfun import compute_mesh
from pysages.colvars import get_periods, wrap
from pysages.grids import build_grid, get_info
from pysages.methods.core import (
    Result,
    SamplingMethod,
    default_getstate,
    default_setstate,
    generalize,
)
from pysages.methods.restraints import CVRestraints, apply_restraints, canonicalize
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple
from pysages.utils import dispatch, first_or_all, linear_solver


# =============================================================================
# Shared Kernel, Support-Gate, & Tiled Helper Functions
# =============================================================================


def _silverman_bandwidth(
    weights: JaxArray,
    sigma_0: JaxArray,
    dims: int,
    sigma_min: JaxArray = None,
) -> JaxArray:
    """Silverman's rule with Kish's effective sample size:

        n_eff = (sum_i w_i)^2 / (sum_i w_i^2)
        sigma = sigma_0 * (n_eff * (d + 2) / 4)^(-1 / (d + 4))
    """
    w = np.maximum(weights, 0.0)
    w_sum = np.sum(w)
    w_sq_sum = np.maximum(np.sum(w ** 2), 1e-30)
    n_eff = np.maximum((w_sum ** 2) / w_sq_sum, 1.0)
    factor = (n_eff * (dims + 2.0) / 4.0) ** (-1.0 / (dims + 4.0))
    sigma = sigma_0 * factor
    if sigma_min is not None:
        sigma = np.maximum(sigma, sigma_min)
    return sigma


def _physical_mesh(grid) -> JaxArray:
    """Return physical grid coordinates of shape `(M, d)` from a PySAGES `Grid`."""
    return compute_mesh(grid)


def _apply_optional_restraints(xi: JaxArray, restraints) -> JaxArray:
    """Evaluate harmonic boundary restraint forces on `xi` if `restraints` is specified."""
    if restraints is None:
        return np.zeros(xi.size, dtype=xi.dtype)
    lo, hi, kl, kh = restraints
    return apply_restraints(lo, hi, kl, kh, xi.reshape(-1))


def _soft_support_gate(m_val: JaxArray, tau_0: float) -> JaxArray:
    """Smooth C^infinity support factor `(1 - exp(-M / tau_0)) / M` avoiding `0 / 0` as `M -> 0`."""
    m_safe = np.maximum(m_val, 1e-12)
    return np.where(
        m_val > 1e-6,
        (1.0 - np.exp(-m_val / tau_0)) / m_safe,
        (1.0 - 0.5 * m_val / tau_0) / tau_0,
    )


def _in_bounds_mask(pts: JaxArray, restraints, grid, periods: JaxArray) -> JaxArray:
    """Return float64 mask `(N,)` equal to `1.0` where `pts` lies inside non-periodic bounds."""
    if restraints is not None:
        lo, hi, _, _ = restraints
        finite_lo = np.isfinite(lo)
        finite_hi = np.isfinite(hi)
        ok_lo = np.where(finite_lo, pts >= lo, True)
        ok_hi = np.where(finite_hi, pts <= hi, True)
        return np.all(ok_lo & ok_hi, axis=-1).astype(np.float64)
    if grid is not None:
        non_per = np.isinf(periods)
        lo = np.asarray(grid.lower, dtype=np.float64).reshape(1, -1)
        hi = np.asarray(grid.upper, dtype=np.float64).reshape(1, -1)
        ok = np.where(non_per.reshape(1, -1), (pts >= lo) & (pts <= hi), True)
        return np.all(ok, axis=-1).astype(np.float64)
    return np.ones(pts.shape[0], dtype=np.float64)


def _fit_energy_gradients(
    centers: JaxArray,
    heights: JaxArray,
    active: JaxArray,
    sigma: JaxArray,
    periods: JaxArray,
) -> JaxArray:
    """Fit ridge-regularized weighted local-linear slopes `G_i = grad V(xi_i)` purely from
    scalar kernel heights `{V_j}` without using momenta or instantaneous forces.
    """
    reg_mat = 0.5 * np.diag(sigma ** 2)
    fit_sig = sigma

    def _fit_one(inp):
        c_i, h_i, act_i = inp
        diff_j = wrap(centers - c_i.reshape(1, -1), periods)
        w_j = active * np.exp(-0.5 * np.sum((diff_j / fit_sig) ** 2, axis=-1))
        dv_j = heights - h_i
        cov = (diff_j.T * w_j) @ diff_j + reg_mat
        rhs = diff_j.T @ (w_j * dv_j)
        g_i = np.linalg.solve(cov, rhs)
        return np.where(act_i > 0.5, g_i, np.zeros_like(g_i))

    return lax_map(_fit_one, (centers, heights, active), batch_size=min(centers.shape[0], 256))


def _kernel_redundancy_tiled(
    centers: JaxArray, active: JaxArray, sigma: JaxArray, periods: JaxArray
) -> JaxArray:
    """Compute `(K_cc @ active) - active` in 256-row tiles without allocating `(k, k, d)`."""

    def _row_sum(c_i):
        diff = wrap(c_i.reshape(1, -1) - centers, periods) / sigma
        k_vec = np.exp(-0.5 * np.sum(diff ** 2, axis=-1))
        return np.sum(k_vec * active)

    return lax_map(_row_sum, centers, batch_size=min(centers.shape[0], 256)) - active


def _batch_coverage_tiled(
    centers: JaxArray,
    active: JaxArray,
    batch_xi: JaxArray,
    sigma: JaxArray,
    periods: JaxArray,
) -> JaxArray:
    """Compute `active @ K(centers, batch_xi)` of shape `(stride,)` in 256-sample tiles."""

    def _pt_cov(xi_s):
        diff = wrap(xi_s.reshape(1, -1) - centers, periods) / sigma
        k_vec = np.exp(-0.5 * np.sum(diff ** 2, axis=-1))
        return np.sum(active * k_vec)

    return lax_map(_pt_cov, batch_xi, batch_size=min(batch_xi.shape[0], 256))


def _dbkmetad_potential_point(
    xi_pt: JaxArray,
    centers: JaxArray,
    heights: JaxArray,
    g_i: JaxArray,
    active: JaxArray,
    sigma: JaxArray,
    periods: JaxArray,
    tau_0: float,
) -> JaxArray:
    """Evaluate online-deposited energy-only local-linear Nadaraya-Watson potential:
        V_n(xi) = g(M(xi)) * sum_{j < m_n} (V_j + G_j . disp_j) * K_sigma(xi, xi_j)
    with smooth bounded displacement `1.5 * sigma * tanh((xi - xi_j) / (1.5 * sigma))`
    and C^infinity unvisited-support gate `g(M) = (1 - exp(-M / tau_0)) / M`.
    """
    diff0 = wrap(xi_pt.reshape(1, -1) - centers, periods)
    diff = diff0 / sigma
    k_vec = active * np.exp(-0.5 * np.sum(diff ** 2, axis=-1))
    m_val = np.sum(k_vec)
    disp = 1.5 * sigma * np.tanh(diff0 / (1.5 * sigma))
    local_lin = heights + np.sum(g_i * disp, axis=-1)
    return _soft_support_gate(m_val, tau_0) * np.sum(local_lin * k_vec)


def _dbkabf_potential_point(
    xi_pt: JaxArray,
    centers: JaxArray,
    c_i: JaxArray,
    f_i: JaxArray,
    active: JaxArray,
    sigma: JaxArray,
    periods: JaxArray,
    tau_0: float,
) -> JaxArray:
    """Evaluate online-deposited local-linear Hermite Nadaraya-Watson potential:
        V_n(xi) = g(M(xi)) * sum_{i < m_n} (C_{i,n} + F_{i,n} . disp_i) * K_sigma(xi, xi_i)
    with bounded smooth tangent displacement `1.5 * sigma * tanh((xi - xi_i) / (1.5 * sigma))`
    having exact unit Jacobian at `xi = xi_i` while preventing tail overshoot.
    """
    diff0 = wrap(xi_pt.reshape(1, -1) - centers, periods)
    diff = diff0 / sigma
    k_vec = active * np.exp(-0.5 * np.sum(diff ** 2, axis=-1))
    m_val = np.sum(k_vec)
    disp = 1.5 * sigma * np.tanh(diff0 / (1.5 * sigma))
    local_lin = c_i + np.sum(f_i * disp, axis=-1)
    return _soft_support_gate(m_val, tau_0) * np.sum(local_lin * k_vec)


class VariationalKernelMethod(SamplingMethod):
    """Base class for the online-deposited variational kernel enhanced sampling methods."""

    __special_args__ = {"grid"}
    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, grid=None, **kwargs):
        super().__init__(cvs, **kwargs)
        self.grid = grid
        raw_rest = kwargs.get("restraints", None)
        if isinstance(raw_rest, tuple) and not isinstance(raw_rest, CVRestraints):
            raw_rest = CVRestraints(*raw_rest)
        self.restraints = canonicalize(raw_rest, cvs)

    def __getstate__(self):
        return (get_info(self.grid), *default_getstate(self))

    def __setstate__(self, state):
        grid_args, args, kwargs = state
        args["grid"] = build_grid(*grid_args)
        default_setstate(self, (args, kwargs))


# =============================================================================
# Method 1: Online-Deposition Batch-Updated Kernel Metadynamics (`DBKMetaD`)
# =============================================================================


class DBKMetaDState(NamedTuple):
    """State for Deposited Batch Kernel Metadynamics (`DBKMetaD`)."""

    xi: JaxArray
    bias: JaxArray
    centers: JaxArray
    heights: JaxArray
    G_i: JaxArray
    sigma: JaxArray
    z_lambda: JaxArray
    batch_xi: JaxArray
    n_kernels: JaxArray
    ncalls: int = 0

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class DBKMetaD(VariationalKernelMethod):
    """Deposited Batch Kernel Metadynamics (`DBKMetaD`).

    Walker-free, grid-free online-deposition batch kernel method:
    - Collects CV configurations in batches of length `stride`.
    - Each batch update selects one `xi` from `batch_xi` (the least-covered in-bounds
      configuration) to deposit one new Gaussian kernel, updates all active kernel
      heights in batch with self-consistent partition-function normalization `z_lambda`,
      and reconstructs local-linear energy slopes `G_i` purely from scalar heights `{V_j}`.
    """

    def __init__(
        self,
        cvs,
        sigma,
        stride=500,
        max_kernels=None,
        alpha_b=None,
        tau_0=0.15,
        deltaT=1500.0,
        T=300.0,
        kB=0.008314462618,
        sigma_min=None,
        **kwargs,
    ):
        nk_alias = kwargs.pop("nkernels", None)
        if max_kernels is None:
            max_kernels = nk_alias if nk_alias is not None else 2000
        super().__init__(cvs, **kwargs)
        self.sigma = sigma
        self.stride = int(stride)
        self.max_kernels = int(max_kernels)
        self.alpha_b = None if alpha_b is None else float(alpha_b)
        self.tau_0 = float(tau_0)
        self.deltaT = float(deltaT)
        self.T = float(T)
        self.kB = float(kB)
        self.sigma_min = sigma_min

    def build(self, snapshot, helpers, *args, **kwargs):
        return _dbkmetad(self, snapshot, helpers)


def _dbkmetad(method: DBKMetaD, snapshot, helpers):
    cv = method.cv
    stride = method.stride
    max_kernels = method.max_kernels
    alpha_0 = float(stride) if method.alpha_b is None else method.alpha_b
    tau_0 = method.tau_0
    kT = method.kB * method.T
    beta = 1.0 / kT
    lam = method.deltaT / (method.T + method.deltaT)
    periods = get_periods(method.cvs)
    restraints = method.restraints
    dims = periods.size
    natoms = np.size(snapshot.positions, 0)

    grad_v_point = grad(_dbkmetad_potential_point, argnums=0)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()), dtype=np.float64)
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        centers = np.zeros((max_kernels, dims), dtype=np.float64)
        heights = np.zeros(max_kernels, dtype=np.float64)
        G_i = np.zeros((max_kernels, dims), dtype=np.float64)
        z_lambda = np.asarray(1.0, dtype=np.float64)
        batch_xi = np.repeat(xi.reshape(1, dims), stride, axis=0)
        n_kernels = np.asarray(0, dtype=np.int32)
        return DBKMetaDState(
            xi, bias, centers, heights, G_i, sigma_0, z_lambda, batch_xi, n_kernels, 0
        )

    def update_kernels(state: DBKMetaDState):
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        sigma_min = (
            np.broadcast_to(np.asarray(method.sigma_min, dtype=np.float64), (dims,))
            if method.sigma_min is not None
            else None
        )
        idx_arange = np.arange(max_kernels)
        old_active = (idx_arange < state.n_kernels).astype(np.float64)

        # 1. Choose the least-covered in-bounds configuration in `batch_xi` to deposit
        in_b = _in_bounds_mask(state.batch_xi, restraints, method.grid, periods)
        batch_cov = _batch_coverage_tiled(
            state.centers, old_active, state.batch_xi, state.sigma, periods
        ) + (1.0 - in_b) * 1e6
        best_s = np.argmin(batch_cov - 1e-9 * np.arange(stride))
        xi_new = state.batch_xi[best_s]

        # Continuous initialization from current surface V_n(xi_new) and grad V_n(xi_new)
        v_init = np.where(
            state.n_kernels > 0,
            _dbkmetad_potential_point(
                xi_new,
                state.centers,
                state.heights,
                state.G_i,
                old_active,
                state.sigma,
                periods,
                tau_0,
            ),
            0.0,
        )
        g_init = np.where(
            state.n_kernels > 0,
            grad_v_point(
                xi_new,
                state.centers,
                state.heights,
                state.G_i,
                old_active,
                state.sigma,
                periods,
                tau_0,
            ),
            np.zeros(dims, dtype=np.float64),
        )

        # Target index: append at n_kernels, or overwrite most redundant kernel if full
        redundancy = _kernel_redundancy_tiled(state.centers, old_active, state.sigma, periods)
        redundant_idx = np.argmax(np.where(old_active > 0.5, redundancy, -1.0))
        write_idx = np.where(state.n_kernels < max_kernels, state.n_kernels, redundant_idx)
        new_n_kernels = np.minimum(state.n_kernels + 1, max_kernels)
        new_active = (idx_arange < new_n_kernels).astype(np.float64)

        centers_1 = state.centers.at[write_idx].set(xi_new)
        heights_1 = state.heights.at[write_idx].set(v_init)
        G_1 = state.G_i.at[write_idx].set(g_init)

        # 2. Batch update of all active kernel heights with partition-function tempering z_lambda
        v_at_centers = lax_map(
            lambda c_i: _dbkmetad_potential_point(
                c_i, centers_1, heights_1, G_1, new_active, state.sigma, periods, tau_0
            ),
            centers_1,
            batch_size=min(max_kernels, 256),
        )

        half_sig = state.sigma / 2.0
        inv_stride = 1.0 / float(stride)
        rho_b = lax_map(
            lambda c_i: np.sum(
                np.exp(
                    -0.5
                    * np.sum(
                        (wrap(c_i.reshape(1, -1) - state.batch_xi, periods) / half_sig) ** 2,
                        axis=-1,
                    )
                )
                * in_b
            )
            * inv_stride,
            centers_1,
            batch_size=min(max_kernels, 256),
        )
        exponent = np.clip(beta * (lam * v_at_centers - heights_1), -80.0, 80.0)
        alpha_eff = alpha_0 / state.z_lambda
        delta_h = (1.0 / beta) * np.log1p(alpha_eff * np.exp(exponent) * rho_b)
        new_heights = np.where(new_active > 0.5, heights_1 + delta_h, 0.0)

        # 3. Self-consistent partition-function normalization Z_{lambda,n+1} (grid-free)
        warmup = 1.0 - np.exp(-new_n_kernels / 40.0)
        h_act_pos = np.where(new_active > 0.5, np.maximum(new_heights, 0.0), 0.0)
        v_mean_active = np.sum(h_act_pos) / np.maximum(np.sum(new_active), 1.0)
        new_z_lambda = np.exp(np.clip(beta * lam * 0.70 * warmup * v_mean_active, 0.0, 60.0))

        # 4. Optional Silverman bandwidth adaptation over active heights
        if sigma_min is not None:
            h_min = np.min(np.where(new_active > 0.5, new_heights, 1e30))
            w_act = np.where(new_active > 0.5, np.maximum(new_heights - h_min, 1e-8), 0.0)
            new_sigma = _silverman_bandwidth(w_act, sigma_0, dims, sigma_min=sigma_min)
        else:
            new_sigma = state.sigma

        # 5. Reconstruct energy-only local-linear slopes G_i from updated scalar heights
        new_G_i = _fit_energy_gradients(centers_1, new_heights, new_active, new_sigma, periods)

        return centers_1, new_heights, new_G_i, new_sigma, new_z_lambda, new_n_kernels

    def skip_update(state: DBKMetaDState):
        return (
            state.centers,
            state.heights,
            state.G_i,
            state.sigma,
            state.z_lambda,
            state.n_kernels,
        )

    def update(state: DBKMetaDState, data):
        xi, Jxi = cv(data)
        ncalls = state.ncalls + 1
        slot = state.ncalls % stride
        batch_xi = state.batch_xi.at[slot].set(xi.flatten())

        state_with_sample = DBKMetaDState(
            xi,
            state.bias,
            state.centers,
            state.heights,
            state.G_i,
            state.sigma,
            state.z_lambda,
            batch_xi,
            state.n_kernels,
            ncalls,
        )
        should_update = ncalls % stride == 0
        centers, heights, G_i, sigma, z_lambda, n_kernels = cond(
            should_update, update_kernels, skip_update, state_with_sample
        )

        active = (np.arange(max_kernels) < n_kernels).astype(np.float64)
        grad_v_xi = lam * grad_v_point(
            xi.flatten(), centers, heights, G_i, active, sigma, periods, tau_0
        )
        rest_force = _apply_optional_restraints(xi, restraints)
        force = grad_v_xi + rest_force
        bias = (-Jxi.T @ force).reshape(state.bias.shape)

        return DBKMetaDState(
            xi, bias, centers, heights, G_i, sigma, z_lambda, batch_xi, n_kernels, ncalls
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


# =============================================================================
# Method 2: First-Order Online-Deposition Batch Kernel Metadynamics (`FOKMetaD`)
# =============================================================================


class FOKMetaDState(NamedTuple):
    """State for First-Order Online-Deposition Batch Kernel Metadynamics (`FOKMetaD`)."""

    xi: JaxArray
    bias: JaxArray
    centers: JaxArray
    C_i: JaxArray
    G_i: JaxArray
    sigma: JaxArray
    z_lambda: JaxArray
    batch_xi: JaxArray
    n_kernels: JaxArray
    ncalls: int = 0

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class FOKMetaD(VariationalKernelMethod):
    """First-Order Online-Deposition Batch Kernel Metadynamics (`FOKMetaD`).

    Variant of `DBKMetaD` in which the kernel coefficient update replaces the
    logarithmic increment `(1 / beta) * log(1 + u_i)` by its first-order linear
    expansion:
        C_{n+1}(xi_i) = C_n(xi_i)
                        + (1 / beta) * (rho_b(xi_i) / Z_{lambda,n})
                          * exp(beta * (lambda * V_n(xi_i) - C_n(xi_i)))
    bounded within the Maclaurin convergence trust region `u_i <= u_max` during
    initial transient steps while retaining `DBKMetaD`'s sparsity-guided kernel
    deposition, smooth support gate `g(M)`, partition-function normalization
    `z_lambda`, and energy-only local-linear gradient reconstruction `G_i`.
    """

    def __init__(
        self,
        cvs,
        sigma,
        stride=500,
        max_kernels=None,
        alpha_b=None,
        u_max=1.0,
        tau_0=0.15,
        deltaT=1500.0,
        T=300.0,
        kB=0.008314462618,
        sigma_min=None,
        **kwargs,
    ):
        nk_alias = kwargs.pop("nkernels", None)
        if max_kernels is None:
            max_kernels = nk_alias if nk_alias is not None else 2000
        super().__init__(cvs, **kwargs)
        self.sigma = sigma
        self.stride = int(stride)
        self.max_kernels = int(max_kernels)
        self.alpha_b = None if alpha_b is None else float(alpha_b)
        self.u_max = None if u_max is None else float(u_max)
        self.tau_0 = float(tau_0)
        self.deltaT = float(deltaT)
        self.T = float(T)
        self.kB = float(kB)
        self.sigma_min = sigma_min

    def build(self, snapshot, helpers, *args, **kwargs):
        return _fokmetad(self, snapshot, helpers)


def _fokmetad(method: FOKMetaD, snapshot, helpers):
    cv = method.cv
    stride = method.stride
    max_kernels = method.max_kernels
    alpha_0 = float(stride) if method.alpha_b is None else method.alpha_b
    u_max = method.u_max
    tau_0 = method.tau_0
    kT = method.kB * method.T
    beta = 1.0 / kT
    lam = method.deltaT / (method.T + method.deltaT)
    periods = get_periods(method.cvs)
    restraints = method.restraints
    dims = periods.size
    natoms = np.size(snapshot.positions, 0)

    grad_v_point = grad(_dbkmetad_potential_point, argnums=0)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()), dtype=np.float64)
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        centers = np.zeros((max_kernels, dims), dtype=np.float64)
        C_i = np.zeros(max_kernels, dtype=np.float64)
        G_i = np.zeros((max_kernels, dims), dtype=np.float64)
        z_lambda = np.asarray(1.0, dtype=np.float64)
        batch_xi = np.repeat(xi.reshape(1, dims), stride, axis=0)
        n_kernels = np.asarray(0, dtype=np.int32)
        return FOKMetaDState(
            xi, bias, centers, C_i, G_i, sigma_0, z_lambda, batch_xi, n_kernels, 0
        )

    def update_kernels(state: FOKMetaDState):
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        sigma_min = (
            np.broadcast_to(np.asarray(method.sigma_min, dtype=np.float64), (dims,))
            if method.sigma_min is not None
            else None
        )
        idx_arange = np.arange(max_kernels)
        old_active = (idx_arange < state.n_kernels).astype(np.float64)

        # 1. Choose the least-covered in-bounds configuration in `batch_xi` to deposit
        in_b = _in_bounds_mask(state.batch_xi, restraints, method.grid, periods)
        batch_cov = _batch_coverage_tiled(
            state.centers, old_active, state.batch_xi, state.sigma, periods
        ) + (1.0 - in_b) * 1e6
        best_s = np.argmin(batch_cov - 1e-9 * np.arange(stride))
        xi_new = state.batch_xi[best_s]

        # Continuous initialization from current surface V_n(xi_new) and grad V_n(xi_new)
        c_init = np.where(
            state.n_kernels > 0,
            _dbkmetad_potential_point(
                xi_new,
                state.centers,
                state.C_i,
                state.G_i,
                old_active,
                state.sigma,
                periods,
                tau_0,
            ),
            0.0,
        )
        g_init = np.where(
            state.n_kernels > 0,
            grad_v_point(
                xi_new,
                state.centers,
                state.C_i,
                state.G_i,
                old_active,
                state.sigma,
                periods,
                tau_0,
            ),
            np.zeros(dims, dtype=np.float64),
        )

        redundancy = _kernel_redundancy_tiled(state.centers, old_active, state.sigma, periods)
        redundant_idx = np.argmax(np.where(old_active > 0.5, redundancy, -1.0))
        write_idx = np.where(state.n_kernels < max_kernels, state.n_kernels, redundant_idx)
        new_n_kernels = np.minimum(state.n_kernels + 1, max_kernels)
        new_active = (idx_arange < new_n_kernels).astype(np.float64)

        centers_1 = state.centers.at[write_idx].set(xi_new)
        C_1 = state.C_i.at[write_idx].set(c_init)
        G_1 = state.G_i.at[write_idx].set(g_init)

        # 2. First-order batch update of all active kernel coefficients C_i
        v_at_centers = lax_map(
            lambda c_i: _dbkmetad_potential_point(
                c_i, centers_1, C_1, G_1, new_active, state.sigma, periods, tau_0
            ),
            centers_1,
            batch_size=min(max_kernels, 256),
        )

        half_sig = state.sigma / 2.0
        inv_stride = 1.0 / float(stride)
        rho_b = lax_map(
            lambda c_i: np.sum(
                np.exp(
                    -0.5
                    * np.sum(
                        (wrap(c_i.reshape(1, -1) - state.batch_xi, periods) / half_sig) ** 2,
                        axis=-1,
                    )
                )
                * in_b
            )
            * inv_stride,
            centers_1,
            batch_size=min(max_kernels, 256),
        )
        exponent = np.clip(beta * (lam * v_at_centers - C_1), -80.0, 80.0)
        alpha_eff = alpha_0 / state.z_lambda
        u_linear = alpha_eff * rho_b * np.exp(exponent)
        if u_max is not None:
            u_linear = np.minimum(u_linear, u_max)
        delta_c = (1.0 / beta) * u_linear
        new_C_i = np.where(new_active > 0.5, C_1 + delta_c, 0.0)

        # 3. Self-consistent partition-function normalization Z_{lambda,n+1} (grid-free)
        warmup = 1.0 - np.exp(-new_n_kernels / 40.0)
        c_act_pos = np.where(new_active > 0.5, np.maximum(new_C_i, 0.0), 0.0)
        v_mean_active = np.sum(c_act_pos) / np.maximum(np.sum(new_active), 1.0)
        new_z_lambda = np.exp(np.clip(beta * lam * 0.70 * warmup * v_mean_active, 0.0, 60.0))

        # 4. Optional Silverman bandwidth adaptation over active coefficients
        if sigma_min is not None:
            c_min = np.min(np.where(new_active > 0.5, new_C_i, 1e30))
            w_act = np.where(new_active > 0.5, np.maximum(new_C_i - c_min, 1e-8), 0.0)
            new_sigma = _silverman_bandwidth(w_act, sigma_0, dims, sigma_min=sigma_min)
        else:
            new_sigma = state.sigma

        # 5. Reconstruct energy-only local-linear slopes G_i from updated coefficients
        new_G_i = _fit_energy_gradients(centers_1, new_C_i, new_active, new_sigma, periods)

        return centers_1, new_C_i, new_G_i, new_sigma, new_z_lambda, new_n_kernels

    def skip_update(state: FOKMetaDState):
        return (
            state.centers,
            state.C_i,
            state.G_i,
            state.sigma,
            state.z_lambda,
            state.n_kernels,
        )

    def update(state: FOKMetaDState, data):
        xi, Jxi = cv(data)
        ncalls = state.ncalls + 1
        slot = state.ncalls % stride
        batch_xi = state.batch_xi.at[slot].set(xi.flatten())

        state_with_sample = FOKMetaDState(
            xi,
            state.bias,
            state.centers,
            state.C_i,
            state.G_i,
            state.sigma,
            state.z_lambda,
            batch_xi,
            state.n_kernels,
            ncalls,
        )
        should_update = ncalls % stride == 0
        centers, C_i, G_i, sigma, z_lambda, n_kernels = cond(
            should_update, update_kernels, skip_update, state_with_sample
        )

        active = (np.arange(max_kernels) < n_kernels).astype(np.float64)
        grad_v_xi = lam * grad_v_point(
            xi.flatten(), centers, C_i, G_i, active, sigma, periods, tau_0
        )
        rest_force = _apply_optional_restraints(xi, restraints)
        force = grad_v_xi + rest_force
        bias = (-Jxi.T @ force).reshape(state.bias.shape)

        return FOKMetaDState(
            xi, bias, centers, C_i, G_i, sigma, z_lambda, batch_xi, n_kernels, ncalls
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


# =============================================================================
# Method 3: Online-Deposition Batch-Updated Kernel ABF (`DBKABF`)
# =============================================================================


class DBKABFState(NamedTuple):
    """State for Deposited Batch Kernel Adaptive Biasing Force (`DBKABF`)."""

    xi: JaxArray
    bias: JaxArray
    centers: JaxArray
    C_i: JaxArray
    F_i: JaxArray
    N_i: JaxArray
    sigma: JaxArray
    z_lambda: JaxArray
    force: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    batch_xi: JaxArray
    batch_force: JaxArray
    n_kernels: JaxArray
    ncalls: int = 0

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class DBKABF(VariationalKernelMethod):
    """Deposited Batch Kernel Adaptive Biasing Force (`DBKABF`).

    Walker-free, grid-free online-deposition batch kernel adaptive biasing force:
    - Collects CV configurations `batch_xi` and instantaneous generalized forces `batch_force`
      (evaluated via first-order time derivatives of generalized momenta `W . p`) in batches
      of length `stride`.
    - Each batch update selects one `xi` from `batch_xi` (the least-covered in-bounds
      configuration) to deposit one new kernel and updates both scalar energy offsets `C_i`
      (with self-consistent partition-function normalization `z_lambda`) and conditional
      mean generalized force vectors `F_i` across all active kernels in batch.
    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(
        self,
        cvs,
        sigma,
        stride=500,
        max_kernels=None,
        N=10,
        alpha_b=None,
        tau_0=0.15,
        deltaT=1500.0,
        T=300.0,
        kB=0.008314462618,
        sigma_min=None,
        use_pinv=False,
        **kwargs,
    ):
        nk_alias = kwargs.pop("nkernels", None)
        if max_kernels is None:
            max_kernels = nk_alias if nk_alias is not None else 2000
        super().__init__(cvs, **kwargs)
        self.sigma = sigma
        self.stride = int(stride)
        self.max_kernels = int(max_kernels)
        self.N = int(N)
        self.alpha_b = None if alpha_b is None else float(alpha_b)
        self.tau_0 = float(tau_0)
        self.deltaT = float(deltaT)
        self.T = float(T)
        self.kB = float(kB)
        self.sigma_min = sigma_min
        self.use_pinv = bool(use_pinv)

    def build(self, snapshot, helpers, *args, **kwargs):
        return _dbkabf(self, snapshot, helpers)


def _dbkabf(method: DBKABF, snapshot, helpers):
    cv = method.cv
    stride = method.stride
    max_kernels = method.max_kernels
    n_ramp = float(method.N)
    alpha_0 = float(stride) if method.alpha_b is None else method.alpha_b
    tau_0 = method.tau_0
    kT = method.kB * method.T
    beta = 1.0 / kT
    lam = method.deltaT / (method.T + method.deltaT)
    periods = get_periods(method.cvs)
    restraints = method.restraints
    dims = periods.size

    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)
    tsolve = linear_solver(method.use_pinv)
    query, dimensionality, to_force_units = helpers
    grad_v_point = grad(_dbkabf_potential_point, argnums=0)

    def initialize():
        xi, _ = cv(query(snapshot))
        bias = np.zeros((natoms, dimensionality()), dtype=np.float64)
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        centers = np.zeros((max_kernels, dims), dtype=np.float64)
        C_i = np.zeros(max_kernels, dtype=np.float64)
        F_i = np.zeros((max_kernels, dims), dtype=np.float64)
        N_i = np.zeros(max_kernels, dtype=np.float64)
        z_lambda = np.asarray(1.0, dtype=np.float64)
        force = np.zeros(dims, dtype=np.float64)
        Wp = np.zeros(dims, dtype=np.float64)
        Wp_ = np.zeros(dims, dtype=np.float64)
        batch_xi = np.repeat(xi.reshape(1, dims), stride, axis=0)
        batch_force = np.zeros((stride, dims), dtype=np.float64)
        n_kernels = np.asarray(0, dtype=np.int32)
        return DBKABFState(
            xi,
            bias,
            centers,
            C_i,
            F_i,
            N_i,
            sigma_0,
            z_lambda,
            force,
            Wp,
            Wp_,
            batch_xi,
            batch_force,
            n_kernels,
            0,
        )

    def update_kernels(state: DBKABFState):
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        sigma_min = (
            np.broadcast_to(np.asarray(method.sigma_min, dtype=np.float64), (dims,))
            if method.sigma_min is not None
            else None
        )
        idx_arange = np.arange(max_kernels)
        old_active = (idx_arange < state.n_kernels).astype(np.float64)

        # 1. Choose the least-covered in-bounds configuration in `batch_xi` to deposit
        in_b = _in_bounds_mask(state.batch_xi, restraints, method.grid, periods)
        batch_cov = _batch_coverage_tiled(
            state.centers, old_active, state.batch_xi, state.sigma, periods
        ) + (1.0 - in_b) * 1e6
        best_s = np.argmin(batch_cov - 1e-9 * np.arange(stride))
        xi_new = state.batch_xi[best_s]

        c_init = np.where(
            state.n_kernels > 0,
            _dbkabf_potential_point(
                xi_new,
                state.centers,
                state.C_i,
                state.F_i,
                old_active,
                state.sigma,
                periods,
                tau_0,
            ),
            0.0,
        )

        redundancy = _kernel_redundancy_tiled(state.centers, old_active, state.sigma, periods)
        redundant_idx = np.argmax(np.where(old_active > 0.5, redundancy, -1.0))
        write_idx = np.where(state.n_kernels < max_kernels, state.n_kernels, redundant_idx)
        new_n_kernels = np.minimum(state.n_kernels + 1, max_kernels)
        new_active = (idx_arange < new_n_kernels).astype(np.float64)

        centers_1 = state.centers.at[write_idx].set(xi_new)
        C_1 = state.C_i.at[write_idx].set(c_init)
        F_1 = state.F_i.at[write_idx].set(np.zeros(dims, dtype=np.float64))
        N_1 = state.N_i.at[write_idx].set(0.0)

        # 2. Batch update of scalar offsets C_i and generalized force vectors F_i
        v_at_centers = lax_map(
            lambda c_i: _dbkabf_potential_point(
                c_i, centers_1, C_1, F_1, new_active, state.sigma, periods, tau_0
            ),
            centers_1,
            batch_size=min(max_kernels, 256),
        )
        # Narrow-kernel (sigma / 2) in-bounds density and conditional mean-force accumulation (tiled)
        half_sig = state.sigma / 2.0

        def _abf_batch_stats(c_i):
            diff = wrap(c_i.reshape(1, -1) - state.batch_xi, periods) / half_sig
            k_row = np.exp(-0.5 * np.sum(diff ** 2, axis=-1)) * in_b
            return np.sum(k_row), k_row @ state.batch_force

        dN, dSF = lax_map(_abf_batch_stats, centers_1, batch_size=min(max_kernels, 256))
        rho_b = dN / float(stride)
        exponent = np.clip(beta * (lam * v_at_centers - C_1), -80.0, 80.0)
        alpha_eff = alpha_0 / state.z_lambda
        delta_c = (1.0 / beta) * np.log1p(alpha_eff * np.exp(exponent) * rho_b)
        new_C_i = np.where(new_active > 0.5, C_1 + delta_c, 0.0)

        warmup = 1.0 - np.exp(-new_n_kernels / 40.0)
        c_act_pos = np.where(new_active > 0.5, np.maximum(new_C_i, 0.0), 0.0)
        v_mean_active = np.sum(c_act_pos) / np.maximum(np.sum(new_active), 1.0)
        new_z_lambda = np.exp(np.clip(beta * lam * 0.70 * warmup * v_mean_active, 0.0, 60.0))

        prev_denom = np.where(N_1 > 0.0, np.maximum(N_1, n_ramp), 0.0)
        f_sum = prev_denom[:, None] * F_1 + dSF
        new_N_i = np.where(new_active > 0.5, N_1 + dN, 0.0)
        next_denom = np.maximum(new_N_i, n_ramp)
        new_F_i = np.where(new_active[:, None] > 0.5, f_sum / next_denom[:, None], 0.0)

        if sigma_min is not None:
            c_min = np.min(np.where(new_active > 0.5, new_C_i, 1e30))
            w_act = np.where(new_active > 0.5, np.maximum(new_C_i - c_min, 1e-8), 0.0)
            new_sigma = _silverman_bandwidth(w_act, sigma_0, dims, sigma_min=sigma_min)
        else:
            new_sigma = state.sigma

        return centers_1, new_C_i, new_F_i, new_N_i, new_sigma, new_z_lambda, new_n_kernels

    def skip_update(state: DBKABFState):
        return (
            state.centers,
            state.C_i,
            state.F_i,
            state.N_i,
            state.sigma,
            state.z_lambda,
            state.n_kernels,
        )

    def update(state: DBKABFState, data):
        xi, Jxi = cv(data)
        p = data.momenta
        Wp = tsolve(Jxi, p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        ncalls = state.ncalls + 1

        # Remove total applied CV force from previous step; guard first 2 finite-difference steps
        raw_inst_force = to_force_units(dWp_dt).flatten() + state.force
        inst_force = np.where(ncalls > 2, raw_inst_force, np.zeros_like(state.force))

        slot = state.ncalls % stride
        batch_xi = state.batch_xi.at[slot].set(xi.flatten())
        batch_force = state.batch_force.at[slot].set(inst_force)

        state_with_sample = DBKABFState(
            xi,
            state.bias,
            state.centers,
            state.C_i,
            state.F_i,
            state.N_i,
            state.sigma,
            state.z_lambda,
            state.force,
            Wp,
            state.Wp,
            batch_xi,
            batch_force,
            state.n_kernels,
            ncalls,
        )
        should_update = ncalls % stride == 0
        centers, C_i, F_i, N_i, sigma, z_lambda, n_kernels = cond(
            should_update, update_kernels, skip_update, state_with_sample
        )

        active = (np.arange(max_kernels) < n_kernels).astype(np.float64)
        nw_force = lam * grad_v_point(
            xi.flatten(), centers, C_i, F_i, active, sigma, periods, tau_0
        )
        rest_force = _apply_optional_restraints(xi, restraints)
        total_force = nw_force + rest_force
        bias = (-Jxi.T @ total_force).reshape(state.bias.shape)

        return DBKABFState(
            xi,
            bias,
            centers,
            C_i,
            F_i,
            N_i,
            sigma,
            z_lambda,
            total_force,
            Wp,
            state.Wp,
            batch_xi,
            batch_force,
            n_kernels,
            ncalls,
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


# =============================================================================
# Method 4: Online-Deposition Batch CZAR (`DBCZAR`)
# =============================================================================


class DBCZARState(NamedTuple):
    """State for Deposited Batch Kernel CZAR (`DBCZAR`)."""

    xi: JaxArray
    bias: JaxArray
    centers: JaxArray
    C_i: JaxArray
    G_i: JaxArray
    N_i: JaxArray
    sigma: JaxArray
    z_lambda: JaxArray
    batch_xi: JaxArray
    n_kernels: JaxArray
    ncalls: int = 0

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class DBCZAR(VariationalKernelMethod):
    """Deposited Batch Kernel CZAR (`DBCZAR`).

    Momentum-free variant of `DBKABF` in which the generalized-momentum time
    derivative `d(W . p)/dt` is replaced by a count-weighted online Corrected
    z-Averaged Restraint (CZAR) batch score estimator:
        N_{i,n+1} = N_{i,n} + dN_{i,b}
        G_{n+1}(xi_i) = [ max(N_{i,n}, N_ramp) * G_n(xi_i)
                          + (1 / beta) * grad_xi(dN_{i,b}) + lambda * (dV_n/dxi) * dN_{i,b} ]
                        / max(N_{i,n+1}, N_ramp)
    paired with `sigma / 2` scalar density probing and tempered log-sum-exp
    partition-function normalization `Z_{lambda,n}`, requiring only collective-variable
    configurations (`positions` and `indices`, no atomic momenta).
    """

    def __init__(
        self,
        cvs,
        sigma,
        stride=500,
        max_kernels=None,
        N=25,
        alpha_b=None,
        tau_0=0.15,
        deltaT=1500.0,
        T=300.0,
        kB=0.008314462618,
        sigma_min=None,
        **kwargs,
    ):
        nk_alias = kwargs.pop("nkernels", None)
        if max_kernels is None:
            max_kernels = nk_alias if nk_alias is not None else 2000
        super().__init__(cvs, **kwargs)
        self.sigma = sigma
        self.stride = int(stride)
        self.max_kernels = int(max_kernels)
        self.N = int(N)
        self.alpha_b = None if alpha_b is None else float(alpha_b)
        self.tau_0 = float(tau_0)
        self.deltaT = float(deltaT)
        self.T = float(T)
        self.kB = float(kB)
        self.sigma_min = sigma_min

    def build(self, snapshot, helpers, *args, **kwargs):
        return _dbczar(self, snapshot, helpers)


def _dbczar(method: DBCZAR, snapshot, helpers):
    cv = method.cv
    stride = method.stride
    max_kernels = method.max_kernels
    n_ramp = float(method.N)
    alpha_0 = float(stride) if method.alpha_b is None else method.alpha_b
    tau_0 = method.tau_0
    kT = method.kB * method.T
    beta = 1.0 / kT
    lam = method.deltaT / (method.T + method.deltaT)
    periods = get_periods(method.cvs)
    restraints = method.restraints
    dims = periods.size
    natoms = np.size(snapshot.positions, 0)

    grad_v_point = grad(_dbkabf_potential_point, argnums=0)
    val_and_grad_v = value_and_grad(_dbkabf_potential_point, argnums=0)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()), dtype=np.float64)
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        centers = np.zeros((max_kernels, dims), dtype=np.float64)
        C_i = np.zeros(max_kernels, dtype=np.float64)
        G_i = np.zeros((max_kernels, dims), dtype=np.float64)
        N_i = np.zeros(max_kernels, dtype=np.float64)
        z_lambda = np.asarray(1.0, dtype=np.float64)
        batch_xi = np.repeat(xi.reshape(1, dims), stride, axis=0)
        n_kernels = np.asarray(0, dtype=np.int32)
        return DBCZARState(
            xi, bias, centers, C_i, G_i, N_i, sigma_0, z_lambda, batch_xi, n_kernels, 0
        )

    def update_kernels(state: DBCZARState):
        sigma_0 = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))
        sigma_min = (
            np.broadcast_to(np.asarray(method.sigma_min, dtype=np.float64), (dims,))
            if method.sigma_min is not None
            else None
        )
        idx_arange = np.arange(max_kernels)
        old_active = (idx_arange < state.n_kernels).astype(np.float64)

        # 1. Choose the least-covered in-bounds configuration in `batch_xi` to deposit
        in_b = _in_bounds_mask(state.batch_xi, restraints, method.grid, periods)
        batch_cov = _batch_coverage_tiled(
            state.centers, old_active, state.batch_xi, state.sigma, periods
        ) + (1.0 - in_b) * 1e6
        best_s = np.argmin(batch_cov - 1e-9 * np.arange(stride))
        xi_new = state.batch_xi[best_s]

        # Continuous scalar initialization C_init = V_n(xi_new); zero slope/count on new center
        # (avoids baking support-gate boundary gradients into newly deposited outer centers)
        c_init = np.where(
            state.n_kernels > 0,
            _dbkabf_potential_point(
                xi_new,
                state.centers,
                state.C_i,
                state.G_i,
                old_active,
                state.sigma,
                periods,
                tau_0,
            ),
            0.0,
        )

        redundancy = _kernel_redundancy_tiled(state.centers, old_active, state.sigma, periods)
        redundant_idx = np.argmax(np.where(old_active > 0.5, redundancy, -1.0))
        write_idx = np.where(state.n_kernels < max_kernels, state.n_kernels, redundant_idx)
        new_n_kernels = np.minimum(state.n_kernels + 1, max_kernels)
        new_active = (idx_arange < new_n_kernels).astype(np.float64)

        centers_1 = state.centers.at[write_idx].set(xi_new)
        C_1 = state.C_i.at[write_idx].set(c_init)
        G_1 = state.G_i.at[write_idx].set(np.zeros(dims, dtype=np.float64))
        N_1 = state.N_i.at[write_idx].set(0.0)

        # 2. Evaluate V_n(xi_i) and dV_n(xi_i)/dxi across active centers in fused 256-row tiles
        v_at_centers, grad_v_centers = lax_map(
            lambda c_i: val_and_grad_v(
                c_i, centers_1, C_1, G_1, new_active, state.sigma, periods, tau_0
            ),
            centers_1,
            batch_size=min(max_kernels, 256),
        )

        # 3. Dimension-consistent narrow scalar probe (`sigma / 2`, matching DBKMetaD/FOKMetaD/DBKABF)
        #    and dimension-scaled CZAR score probe `sigma * 2^(-(d - 2)_+ / d)` with single-batch cap
        inv_stride = 1.0 / float(stride)
        half_sig = state.sigma / 2.0
        sig_g = state.sigma * (2.0 ** (-max(dims - 2, 0) / float(dims)))
        sig_g2 = sig_g ** 2
        batch_cap = 0.35 * n_ramp

        def _czar_batch_stats(c_i):
            diff0 = wrap(c_i.reshape(1, -1) - state.batch_xi, periods)
            k_c = np.exp(-0.5 * np.sum((diff0 / half_sig) ** 2, axis=-1)) * in_b
            r_i = np.sum(k_c) * inv_stride

            k_g = np.exp(-0.5 * np.sum((diff0 / sig_g) ** 2, axis=-1)) * in_b
            dn_raw = np.sum(k_g)
            dn_eff = np.minimum(dn_raw, batch_cap)
            scale_g = np.where(dn_raw > 1e-12, dn_eff / dn_raw, 0.0)
            gr_sum = np.sum(-k_g[:, None] * (diff0 / sig_g2), axis=0) * scale_g
            return r_i, dn_eff, gr_sum

        rho_b, dN_g, grad_sum_g = lax_map(
            _czar_batch_stats, centers_1, batch_size=min(max_kernels, 256)
        )

        # Overflow-safe relative Boltzmann weight for scalar C_i update:
        exponent = np.clip(beta * (lam * v_at_centers - C_1), -80.0, 80.0)
        r_weight = (alpha_0 / state.z_lambda) * np.exp(exponent)

        # Update scalar offsets C_i
        delta_c = (1.0 / beta) * np.log1p(r_weight * rho_b)
        new_C_i = np.where(new_active > 0.5, C_1 + delta_c, 0.0)

        # Update gradient vectors G_i via count-weighted online CZAR estimator (retaining cumulative N_i)
        dS_g = (1.0 / beta) * grad_sum_g + lam * grad_v_centers * dN_g[:, None]
        prev_denom = np.where(N_1 > 0.0, np.maximum(N_1, n_ramp), 0.0)
        g_sum = prev_denom[:, None] * G_1 + dS_g
        new_N_i = np.where(new_active > 0.5, N_1 + dN_g, 0.0)
        next_denom = np.maximum(new_N_i, n_ramp)
        new_G_i = np.where(new_active[:, None] > 0.5, g_sum / next_denom[:, None], 0.0)

        # 4. Tempered log-sum-exp partition-function normalization Z_{lambda,n+1}
        warmup = 1.0 - np.exp(-new_n_kernels / 40.0)
        kappa = 0.25 * beta * lam
        c_pos = np.maximum(new_C_i, 0.0)
        u_act = np.where(new_active > 0.5, kappa * c_pos, -1e30)
        u_max_v = np.max(u_act)
        sum_exp = np.sum(np.where(new_active > 0.5, np.exp(u_act - u_max_v), 0.0))
        m_act = np.maximum(np.sum(new_active), 1.0)
        v_lse = (u_max_v + np.log(np.maximum(sum_exp / m_act, 1e-30))) / np.maximum(kappa, 1e-12)
        new_z_lambda = np.exp(np.clip(beta * lam * 0.70 * warmup * v_lse, 0.0, 60.0))

        # 5. Optional Silverman bandwidth adaptation
        if sigma_min is not None:
            c_min = np.min(np.where(new_active > 0.5, new_C_i, 1e30))
            w_act = np.where(new_active > 0.5, np.maximum(new_C_i - c_min, 1e-8), 0.0)
            new_sigma = _silverman_bandwidth(w_act, sigma_0, dims, sigma_min=sigma_min)
        else:
            new_sigma = state.sigma

        return centers_1, new_C_i, new_G_i, new_N_i, new_sigma, new_z_lambda, new_n_kernels

    def skip_update(state: DBCZARState):
        return (
            state.centers,
            state.C_i,
            state.G_i,
            state.N_i,
            state.sigma,
            state.z_lambda,
            state.n_kernels,
        )

    def update(state: DBCZARState, data):
        xi, Jxi = cv(data)
        ncalls = state.ncalls + 1
        slot = state.ncalls % stride
        batch_xi = state.batch_xi.at[slot].set(xi.flatten())

        state_with_sample = DBCZARState(
            xi,
            state.bias,
            state.centers,
            state.C_i,
            state.G_i,
            state.N_i,
            state.sigma,
            state.z_lambda,
            batch_xi,
            state.n_kernels,
            ncalls,
        )
        should_update = ncalls % stride == 0
        centers, C_i, G_i, N_i, sigma, z_lambda, n_kernels = cond(
            should_update, update_kernels, skip_update, state_with_sample
        )

        active = (np.arange(max_kernels) < n_kernels).astype(np.float64)
        nw_force = lam * grad_v_point(
            xi.flatten(), centers, C_i, G_i, active, sigma, periods, tau_0
        )
        rest_force = _apply_optional_restraints(xi, restraints)
        force = nw_force + rest_force
        bias = (-Jxi.T @ force).reshape(state.bias.shape)

        return DBCZARState(
            xi, bias, centers, C_i, G_i, N_i, sigma, z_lambda, batch_xi, n_kernels, ncalls
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


# =============================================================================
# PySAGES `analyze` Dispatches for All Four Online-Deposited Methods
# =============================================================================


@dispatch
def analyze(result: Result[DBKMetaD]):
    """Reconstruct free-energy surface `A(xi) = -V_n(xi)` for `DBKMetaD`."""
    method = result.method
    periods = get_periods(method.cvs)
    tau_0 = method.tau_0
    mesh = _physical_mesh(method.grid) if method.grid is not None else None

    centers_list, heights_list, slopes_list, n_kernels_list, fes_list, fes_fns = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for s in result.states:
        m = max(int(s.n_kernels), 1)
        active = (np.arange(s.centers.shape[0]) < int(s.n_kernels)).astype(np.float64)
        fes_fn = jit(
            lambda pts, c=s.centers, h=s.heights, gi=s.G_i, act=active, sig=s.sigma: lax_map(
                lambda x: -_dbkmetad_potential_point(x, c, h, gi, act, sig, periods, tau_0),
                np.asarray(pts, dtype=np.float64),
                batch_size=min(max(len(pts), 1), 512),
            )
        )
        act_centers = s.centers[:m]
        act_heights = s.heights[:m]
        act_gi = s.G_i[:m]
        eval_pts = mesh if mesh is not None else act_centers
        fe = fes_fn(eval_pts)
        fe = fe - np.min(fe)
        centers_list.append(act_centers)
        heights_list.append(act_heights)
        slopes_list.append(act_gi)
        n_kernels_list.append(int(s.n_kernels))
        fes_list.append(fe)
        fes_fns.append(fes_fn)

    out = {
        "centers": first_or_all(centers_list),
        "heights": first_or_all(heights_list),
        "G_i": first_or_all(slopes_list),
        "n_kernels": first_or_all(n_kernels_list),
        "free_energy": first_or_all(fes_list),
        "fes_fn": first_or_all(fes_fns),
    }
    if mesh is not None:
        out["mesh"] = mesh
    return numpyfy_vals(out)


@dispatch
def analyze(result: Result[FOKMetaD]):
    """Reconstruct free-energy surface `A(xi) = -V_n(xi)` for `FOKMetaD`."""
    method = result.method
    periods = get_periods(method.cvs)
    tau_0 = method.tau_0
    mesh = _physical_mesh(method.grid) if method.grid is not None else None

    centers_list, heights_list, slopes_list, n_kernels_list, fes_list, fes_fns = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for s in result.states:
        m = max(int(s.n_kernels), 1)
        active = (np.arange(s.centers.shape[0]) < int(s.n_kernels)).astype(np.float64)
        fes_fn = jit(
            lambda pts, c=s.centers, ci=s.C_i, gi=s.G_i, act=active, sig=s.sigma: lax_map(
                lambda x: -_dbkmetad_potential_point(x, c, ci, gi, act, sig, periods, tau_0),
                np.asarray(pts, dtype=np.float64),
                batch_size=min(max(len(pts), 1), 512),
            )
        )
        act_centers = s.centers[:m]
        act_ci = s.C_i[:m]
        act_gi = s.G_i[:m]
        eval_pts = mesh if mesh is not None else act_centers
        fe = fes_fn(eval_pts)
        fe = fe - np.min(fe)
        centers_list.append(act_centers)
        heights_list.append(act_ci)
        slopes_list.append(act_gi)
        n_kernels_list.append(int(s.n_kernels))
        fes_list.append(fe)
        fes_fns.append(fes_fn)

    out = {
        "centers": first_or_all(centers_list),
        "heights": first_or_all(heights_list),
        "C_i": first_or_all(heights_list),
        "G_i": first_or_all(slopes_list),
        "n_kernels": first_or_all(n_kernels_list),
        "free_energy": first_or_all(fes_list),
        "fes_fn": first_or_all(fes_fns),
    }
    if mesh is not None:
        out["mesh"] = mesh
    return numpyfy_vals(out)


@dispatch
def analyze(result: Result[DBKABF]):
    """Reconstruct free-energy surface `A(xi) = -V_n(xi)` for `DBKABF`."""
    method = result.method
    periods = get_periods(method.cvs)
    tau_0 = method.tau_0
    mesh = _physical_mesh(method.grid) if method.grid is not None else None

    centers_list, heights_list, forces_list, n_kernels_list, fes_list, fes_fns = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for s in result.states:
        m = max(int(s.n_kernels), 1)
        active = (np.arange(s.centers.shape[0]) < int(s.n_kernels)).astype(np.float64)
        fes_fn = jit(
            lambda pts, c=s.centers, ci=s.C_i, fi=s.F_i, act=active, sig=s.sigma: lax_map(
                lambda x: -_dbkabf_potential_point(x, c, ci, fi, act, sig, periods, tau_0),
                np.asarray(pts, dtype=np.float64),
                batch_size=min(max(len(pts), 1), 512),
            )
        )
        act_centers = s.centers[:m]
        act_ci = s.C_i[:m]
        act_fi = s.F_i[:m]
        eval_pts = mesh if mesh is not None else act_centers
        fe = fes_fn(eval_pts)
        fe = fe - np.min(fe)
        centers_list.append(act_centers)
        heights_list.append(act_ci)
        forces_list.append(act_fi)
        n_kernels_list.append(int(s.n_kernels))
        fes_list.append(fe)
        fes_fns.append(fes_fn)

    out = {
        "centers": first_or_all(centers_list),
        "heights": first_or_all(heights_list),
        "F_i": first_or_all(forces_list),
        "n_kernels": first_or_all(n_kernels_list),
        "free_energy": first_or_all(fes_list),
        "fes_fn": first_or_all(fes_fns),
    }
    if mesh is not None:
        out["mesh"] = mesh
    return numpyfy_vals(out)


@dispatch
def analyze(result: Result[DBCZAR]):
    """Reconstruct free-energy surface `A(xi) = -V_n(xi)` for `DBCZAR`."""
    method = result.method
    periods = get_periods(method.cvs)
    tau_0 = method.tau_0
    mesh = _physical_mesh(method.grid) if method.grid is not None else None

    centers_list, heights_list, slopes_list, counts_list, n_kernels_list, fes_list, fes_fns = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for s in result.states:
        m = max(int(s.n_kernels), 1)
        active = (np.arange(s.centers.shape[0]) < int(s.n_kernels)).astype(np.float64)
        fes_fn = jit(
            lambda pts, c=s.centers, ci=s.C_i, gi=s.G_i, act=active, sig=s.sigma: lax_map(
                lambda x: -_dbkabf_potential_point(x, c, ci, gi, act, sig, periods, tau_0),
                np.asarray(pts, dtype=np.float64),
                batch_size=min(max(len(pts), 1), 512),
            )
        )
        act_centers = s.centers[:m]
        act_ci = s.C_i[:m]
        act_gi = s.G_i[:m]
        act_ni = s.N_i[:m] if hasattr(s, "N_i") else np.zeros(m, dtype=np.float64)
        eval_pts = mesh if mesh is not None else act_centers
        fe = fes_fn(eval_pts)
        fe = fe - np.min(fe)
        centers_list.append(act_centers)
        heights_list.append(act_ci)
        slopes_list.append(act_gi)
        counts_list.append(act_ni)
        n_kernels_list.append(int(s.n_kernels))
        fes_list.append(fe)
        fes_fns.append(fes_fn)

    out = {
        "centers": first_or_all(centers_list),
        "heights": first_or_all(heights_list),
        "C_i": first_or_all(heights_list),
        "G_i": first_or_all(slopes_list),
        "N_i": first_or_all(counts_list),
        "n_kernels": first_or_all(n_kernels_list),
        "free_energy": first_or_all(fes_list),
        "fes_fn": first_or_all(fes_fns),
    }
    if mesh is not None:
        out["mesh"] = mesh
    return numpyfy_vals(out)


# Descriptive aliases
DepositedBatchKernelMetaD = DBKMetaD
FirstOrderKernelMetaD = FOKMetaD
DepositedBatchKernelABF = DBKABF
DepositedBatchCZAR = DBCZAR
