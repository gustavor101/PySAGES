# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Weights Agnostic eXplainablE Distributions (WAXED) collective variables.

Implements physics-informed, training-free, permutationally invariant distribution
collective variables and smooth statistical divergence / functional metric layers:

- :class:`AxialDensityDivergence`: 1D Gaussian-kernel axial density profile divergence.
- :class:`CoordinationDistributionDivergence`: Pairwise radial coordination profile
  divergence with rational, sigmoidal, or Gaussian activations under general triclinic PBC.
- :class:`ElectrostaticPotentialDivergence`: Solvent/environment electrostatic potential
  distribution divergence along a molecular bond axis.
- :class:`DistributionClassifier`: Smooth two-state classifier / interpolator combining
  divergences to two reference states.
"""

import numpy
from jax import nn
from jax import numpy as jnp

from pysages.colvars.coordinates import barycenter
from pysages.colvars.core import (
    AxisCV,
    CollectiveVariable,
    ThreePointCV,
    TwoPointCV,
    _get_nargs,
)

# CODATA 2018 Coulomb constant 1 / (4 * pi * eps_0) in kJ * nm / (mol * e^2)
COULOMB_KJ_NM_MOL = 138.93545764438198

VALID_METRICS = ("kl", "wasserstein", "sobolev", "js", "hellinger", "he", "l2", "l1", "ln")
VALID_ACTIVATIONS = ("rational", "sigmoid", "gaussian")
VALID_CLASSIFIER_MODES = ("ratio", "difference", "softmax")


# ============================================================ #
#   Geometry, PBC MIC, Smooth Norms, and Switching Functions   #
# ============================================================ #


def safe_norm(v, eps: float = 1.0e-12):
    r"""
    Differentiable Euclidean norm :math:`\sqrt{\|\mathbf{v}\|_2^2 + \epsilon^2}`
    that avoids NaN gradients at :math:`\mathbf{v} = \mathbf{0}`.
    """
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + eps * eps)


def _parse_box(box):
    """
    Validate and convert a simulation cell specification into plain NumPy
    ``(H, H_inv)`` matrices of shape ``(3, 3)`` (rows are lattice vectors),
    or ``(None, None)`` when ``box is None``.
    """
    if box is None:
        return None, None

    arr = numpy.asarray(box, dtype=float)
    if arr.ndim == 0:
        if arr <= 0.0:
            raise ValueError("Scalar box length must be strictly positive.")
        H = numpy.diag([float(arr), float(arr), float(arr)])
    elif arr.ndim == 1:
        if arr.shape != (3,) or numpy.any(arr <= 0.0):
            raise ValueError("1D box must have shape (3,) with strictly positive entries.")
        H = numpy.diag(arr)
    elif arr.ndim == 2:
        if arr.shape != (3, 3):
            raise ValueError(f"2D box cell matrix must have shape (3, 3), got {arr.shape}.")
        H = arr.copy()
    else:
        raise ValueError(f"Unsupported box array shape: {arr.shape}.")

    det = float(numpy.linalg.det(H))
    if abs(det) < 1.0e-12:
        raise ValueError(f"Simulation box matrix is singular (det={det}).")

    H_inv = numpy.linalg.inv(H)
    return H, H_inv


def mic_displacement(dr, box=None, box_inv=None):
    """
    Apply the Minimum Image Convention (MIC) for general orthorhombic or
    triclinic cells ``H`` (row vectors are lattice vectors).

    Parameters
    ----------
    dr : jax.Array
        Displacement vectors of shape ``(..., 3)``.
    box : Optional[jax.Array]
        ``(3,)`` orthorhombic lengths or ``(3, 3)`` cell matrix ``H``.
        If ``None``, returns ``dr`` unchanged.
    box_inv : Optional[jax.Array]
        Precomputed inverse cell matrix ``H^{-1}`` of shape ``(3, 3)``.

    Returns
    -------
    jax.Array
        Minimum-image displacement vectors of shape ``(..., 3)``.
    """
    if box is None:
        return dr

    H = jnp.asarray(box)
    if H.ndim == 1:
        s = dr / H
        return (s - jnp.round(s)) * H

    H_inv = jnp.linalg.inv(H) if box_inv is None else jnp.asarray(box_inv)
    s = dr @ H_inv
    s_mic = s - jnp.round(s)
    return s_mic @ H


def _compute_quadrature_dx(bins: numpy.ndarray):
    """
    Compute grid spacing ``dx`` for numerical integration over ``bins``.
    Returns a Python ``float`` for uniform grids and a 1D ``numpy.ndarray``
    for non-uniform grids.
    """
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("`bins` must be a 1D array with at least 2 grid points.")
    diffs = numpy.diff(bins)
    if numpy.any(diffs <= 0.0):
        raise ValueError("`bins` must be strictly increasing.")
    if numpy.allclose(diffs, diffs[0], rtol=1.0e-10, atol=1.0e-12):
        return float(diffs[0])
    return numpy.gradient(bins)


def rational_switching(r, r0, n: float = 6, m: float = 12):
    r"""
    Smooth rational switching function:

    .. math::

        s(r; r_0, n, m) = \frac{1 - (r / r_0)^n}{1 - (r / r_0)^m}

    Uses the exact :math:`C^\infty` identity :math:`1 / (1 + u^n)` whenever
    :math:`m = 2n`, and a 4th-order Taylor expansion around :math:`u = 1` with
    JAX's double-``where`` safe-input pattern for general :math:`(n, m)` to
    guarantee finite, NaN-free first and second derivatives at :math:`r = r_0`.
    """
    n_f = float(n)
    m_f = float(m)
    if n_f <= 0.0 or m_f <= 0.0:
        raise ValueError("Rational switching exponents `n` and `m` must be positive.")

    # Exact C^infty algebraic identity whenever m == 2n
    if abs(m_f - 2.0 * n_f) < 1.0e-12:
        n_int = int(round(n_f))
        if abs(n_f - n_int) < 1.0e-12 and n_int % 2 == 0:
            u2 = (r * r) / (r0 * r0)
            return 1.0 / (1.0 + jnp.power(u2, n_int // 2))
        u = r / r0
        return 1.0 / (1.0 + jnp.power(u, n_f))

    # General (n, m): 4th-order Taylor series around u = 1 with double-where guard
    u = r / r0
    du = u - 1.0
    a1 = (n_f - 1.0) / 2.0
    a2 = (n_f - 1.0) * (n_f - 2.0) / 6.0
    a3 = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) / 24.0
    a4 = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * (n_f - 4.0) / 120.0
    b1 = (m_f - 1.0) / 2.0
    b2 = (m_f - 1.0) * (m_f - 2.0) / 6.0
    b3 = (m_f - 1.0) * (m_f - 2.0) * (m_f - 3.0) / 24.0
    b4 = (m_f - 1.0) * (m_f - 2.0) * (m_f - 3.0) * (m_f - 4.0) / 120.0

    r0_c = 1.0
    r1_c = a1 - b1
    r2_c = a2 - b2 - b1 * r1_c
    r3_c = a3 - b3 - b1 * r2_c - b2 * r1_c
    r4_c = a4 - b4 - b1 * r3_c - b2 * r2_c - b3 * r1_c

    scale = n_f / m_f
    c0 = scale * r0_c
    c1 = scale * r1_c
    c2 = scale * r2_c
    c3 = scale * r3_c
    c4 = scale * r4_c

    near_one = jnp.abs(du) < 2.0e-3
    u_safe = jnp.where(near_one, 1.005, u)
    exact = (1.0 - jnp.power(u_safe, n_f)) / (1.0 - jnp.power(u_safe, m_f))
    taylor = c0 + du * (c1 + du * (c2 + du * (c3 + du * c4)))
    return jnp.where(near_one, taylor, exact)


def sigmoid_switching(r, r0, kappa: float = 10.0):
    r"""
    Overflow-free S-shaped sigmoidal cumulative switching activation:

    .. math::

        \phi_{\mathrm{cum}}(r; r_0) = \frac{1}{1 + \exp[\kappa (r - r_0)]}
    """
    return nn.sigmoid(-kappa * (r - r0))


def gaussian_activation(r, centers, sigma: float, normalized: bool = False):
    r"""
    Gaussian kernel activation:

    .. math::

        \phi_{\mathrm{Gauss}}(r; s_m) = \exp\!\left[-\frac{(r - s_m)^2}{2\sigma^2}\right]

    Optionally divided by :math:`\sigma \sqrt{2\pi}` when ``normalized=True``.
    """
    vals = jnp.exp(-0.5 * ((r - centers) / sigma) ** 2)
    if normalized:
        vals = vals / (sigma * jnp.sqrt(2.0 * jnp.pi))
    return vals


# ============================================================ #
#   Smooth Statistical Divergences & Functional Metrics        #
# ============================================================ #


def _smooth_positive(x, eps: float = 1.0e-12):
    """
    :math:`C^\\infty` strictly positive floor :math:`\\frac{1}{2}(x + \\sqrt{x^2 + \\epsilon^2})`.
    Matches ``x`` to machine precision for ``x >> eps`` and remains strictly
    positive with finite derivatives for all real ``x``.
    """
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


def kl_divergence(p, q, dx, eps: float = 1.0e-12):
    r"""
    Kullback--Leibler divergence:

    .. math::

        D_{\mathrm{KL}}(P \,\|\, Q) = \int P(s) \log\!\left(\frac{P(s)}{Q(s)}\right) ds
    """
    p_s = _smooth_positive(p, eps)
    q_s = _smooth_positive(q, eps)
    return jnp.sum(p_s * jnp.log(p_s / q_s) * dx)


def js_divergence(p, q, dx, eps: float = 1.0e-12):
    r"""
    Jensen--Shannon divergence:

    .. math::

        D_{\mathrm{JS}}(P \,\|\, Q) = \frac{1}{2} D_{\mathrm{KL}}(P \,\|\, M)
        + \frac{1}{2} D_{\mathrm{KL}}(Q \,\|\, M), \quad M = \frac{1}{2}(P + Q)
    """
    p_s = _smooth_positive(p, eps)
    q_s = _smooth_positive(q, eps)
    m = 0.5 * (p_s + q_s)
    return 0.5 * jnp.sum((p_s * jnp.log(p_s / m) + q_s * jnp.log(q_s / m)) * dx)


def hellinger_distance(p, q, dx, eps: float = 1.0e-12):
    r"""
    Squared Hellinger distance:

    .. math::

        H^2(P, Q) = \frac{1}{2} \int \left(\sqrt{P(s)} - \sqrt{Q(s)}\right)^2 ds
    """
    p_s = _smooth_positive(p, eps)
    q_s = _smooth_positive(q, eps)
    diff = jnp.sqrt(p_s) - jnp.sqrt(q_s)
    return 0.5 * jnp.sum((diff * diff) * dx)


def ln_distance(p, q, dx, order: float = 2.0, eps: float = 1.0e-12):
    r"""
    Smooth :math:`L^n` functional norm:

    .. math::

        D_{L^n}(P \,\|\, Q) = \left(\int |P(s) - Q(s)|^n \, ds\right)^{1/n}
    """
    diff = p - q
    n = float(order)
    if n < 1.0:
        raise ValueError(f"`order` must be >= 1.0, got {n}.")
    if abs(n - 2.0) < 1.0e-12:
        return jnp.sqrt(jnp.sum((diff * diff) * dx) + eps * eps)
    if abs(n - 1.0) < 1.0e-12:
        return jnp.sum(jnp.sqrt(diff * diff + eps * eps) * dx)
    integral = jnp.sum(((diff * diff + eps * eps) ** (0.5 * n)) * dx)
    return (integral + eps**n) ** (1.0 / n)


def wasserstein_distance(p, q, dx, order: float = 1.0, eps: float = 1.0e-12):
    r"""
    Smooth 1D :math:`p`-Wasserstein / CDF :math:`L^p` optimal transport distance:

    .. math::

        W_p(P, Q) = \left(\int |F_P(s) - F_Q(s)|^p \, ds\right)^{1/p}

    where :math:`F_P(s) = \int_{-\infty}^s P(s') ds'` and
    :math:`F_Q(s) = \int_{-\infty}^s Q(s') ds'`. Exact 1D optimal transport
    distance for :math:`p = 1` on normalized probability densities.
    """
    p_ord = float(order)
    if p_ord < 1.0:
        raise ValueError(f"Wasserstein `order` must be >= 1.0, got {p_ord}.")
    cdf_p = jnp.cumsum(p * dx)
    cdf_q = jnp.cumsum(q * dx)
    diff = cdf_p - cdf_q
    if abs(p_ord - 1.0) < 1.0e-12:
        return jnp.sum(jnp.sqrt(diff * diff + eps * eps) * dx)
    if abs(p_ord - 2.0) < 1.0e-12:
        return jnp.sqrt(jnp.sum((diff * diff) * dx) + eps * eps)
    integral = jnp.sum(((diff * diff + eps * eps) ** (0.5 * p_ord)) * dx)
    return (integral + eps**p_ord) ** (1.0 / p_ord)


def sobolev_distance(p, q, dx, grid=None, grad_weight: float = 1.0, eps: float = 1.0e-12):
    r"""
    Sobolev :math:`H^{1,2}` functional metric:

    .. math::

        D_{H^{1,2}}(P \,\|\, Q) = \left(\int \left[|P(s) - Q(s)|^2
        + \alpha |\nabla_s P(s) - \nabla_s Q(s)|^2\right] ds\right)^{1/2}
    """
    diff = p - q
    ddiff = jnp.gradient(diff, dx) if grid is None else jnp.gradient(diff, grid)
    integrand = diff * diff + grad_weight * (ddiff * ddiff)
    return jnp.sqrt(jnp.sum(integrand * dx) + eps * eps)


def compute_divergence(
    p,
    q,
    dx,
    metric: str = "kl",
    order: float = None,
    grad_weight: float = 1.0,
    grid=None,
    eps: float = 1.0e-12,
):
    """
    Unified WAXED dimensionality-reduction operator :math:`D(P \\,\\|\\, Q)`.

    Parameters
    ----------
    p : jax.Array
        Instantaneous distribution evaluated at grid nodes, shape ``(M,)``.
    q : jax.Array
        Reference distribution evaluated at the same grid nodes, shape ``(M,)``.
    dx : Union[float, jax.Array]
        Uniform grid spacing or 1D array of quadrature weights of shape ``(M,)``.
    metric : str
        One of ``'kl'``, ``'wasserstein'``, ``'sobolev'``, ``'js'``,
        ``'hellinger'`` (or ``'he'``), ``'l2'``, ``'l1'``, ``'ln'``.
    order : Optional[float]
        Exponent :math:`p` for ``'wasserstein'`` (default ``1.0``) or :math:`n`
        for ``'ln'`` (default ``2.0``).
    grad_weight : float
        Weight :math:`\\alpha` on the spatial gradient term in ``'sobolev'``.
    grid : Optional[jax.Array]
        Optional 1D grid coordinates for non-uniform spatial differentiation
        in ``'sobolev'``.
    eps : float
        Small smoothing parameter ensuring :math:`C^\\infty` differentiability.

    Returns
    -------
    jax.Array
        Scalar divergence / distance value.
    """
    key = metric.lower()
    if key == "kl":
        return kl_divergence(p, q, dx, eps=eps)
    if key == "js":
        return js_divergence(p, q, dx, eps=eps)
    if key in ("hellinger", "he"):
        return hellinger_distance(p, q, dx, eps=eps)
    if key == "l2":
        return ln_distance(p, q, dx, order=2.0, eps=eps)
    if key == "l1":
        return ln_distance(p, q, dx, order=1.0, eps=eps)
    if key == "ln":
        n_ord = 2.0 if order is None else float(order)
        return ln_distance(p, q, dx, order=n_ord, eps=eps)
    if key == "wasserstein":
        p_ord = 1.0 if order is None else float(order)
        return wasserstein_distance(p, q, dx, order=p_ord, eps=eps)
    if key == "sobolev":
        return sobolev_distance(p, q, dx, grid=grid, grad_weight=grad_weight, eps=eps)
    raise ValueError(f"Unknown metric '{metric}'. Valid options are: {VALID_METRICS}.")


# ============================================================ #
#   1. Axial 1D Density Distribution WAXED CV                  #
# ============================================================ #


def axial_density_profile(
    positions,
    axis: int,
    box_length: float,
    num_bins: int = 100,
    bandwidth: float = 0.5,
    periodic: bool = True,
    center_com: bool = True,
    weights=None,
    normalize: bool = True,
):
    r"""
    Compute a smooth, permutationally invariant 1D axial density profile along
    Cartesian axis ``axis`` using Gaussian activations:

    .. math::

        p(s_m; \mathbf{R}) = \frac{1}{\mathcal{Z}} \sum_{i=1}^{N} w_i \,
        \phi_{\mathrm{Gauss}}(x_i; s_m, \sigma)

    When ``periodic=True``, centering uses the periodic circular mean
    (Bai--Breen method) and the Gaussian kernel is evaluated over wrapped
    periodic images :math:`k \in \{-2, -1, 0, 1, 2\}`, ensuring exact
    invariance under per-atom lattice shifts :math:`x_i \to x_i + n_i L` and
    global translations :math:`x_i \to x_i + t`, with :math:`C^\infty`
    smoothness across the periodic boundary.

    Returns
    -------
    grid : jax.Array
        Cell-center grid coordinates of shape ``(num_bins,)`` on ``[-L/2, L/2)``.
    density : jax.Array
        Evaluated density profile of shape ``(num_bins,)``.
    """
    x = positions[:, axis]
    dx = box_length / num_bins
    grid = (-0.5 + (jnp.arange(num_bins) + 0.5) / num_bins) * box_length

    if weights is not None:
        w = jnp.asarray(weights)
        w_norm = w / jnp.sum(w)
    else:
        w = None
        w_norm = None

    if center_com:
        if periodic:
            theta = (2.0 * jnp.pi / box_length) * x
            if w_norm is not None:
                c_bar = jnp.sum(w_norm * jnp.cos(theta))
                s_bar = jnp.sum(w_norm * jnp.sin(theta))
            else:
                c_bar = jnp.mean(jnp.cos(theta))
                s_bar = jnp.mean(jnp.sin(theta))
            deg = (c_bar * c_bar + s_bar * s_bar) < 1.0e-24
            c_safe = jnp.where(deg, 1.0, c_bar)
            s_safe = jnp.where(deg, 0.0, s_bar)
            com = jnp.where(deg, 0.0, (box_length / (2.0 * jnp.pi)) * jnp.arctan2(s_safe, c_safe))
            x = x - com
        else:
            com = jnp.sum(w_norm * x) if w_norm is not None else jnp.mean(x)
            x = x - com

    d = x[:, None] - grid[None, :]
    norm_const = 1.0 / (bandwidth * jnp.sqrt(2.0 * jnp.pi))
    if periodic:
        d_mic = d - box_length * jnp.round(d / box_length)
        shifts = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * box_length
        d_shifted = d_mic[:, :, None] + shifts[None, None, :]
        kernels = norm_const * jnp.sum(jnp.exp(-0.5 * (d_shifted / bandwidth) ** 2), axis=-1)
    else:
        kernels = norm_const * jnp.exp(-0.5 * (d / bandwidth) ** 2)

    if w is not None:
        density = jnp.sum(w[:, None] * kernels, axis=0)
    else:
        density = jnp.sum(kernels, axis=0)

    if normalize:
        total = jnp.sum(density) * dx + 1.0e-24
        density = density / total

    return grid, density


class AxialDensityDivergence(AxisCV):
    r"""
    WAXED 1D axial density distribution divergence collective variable.

    Constructs a continuous, permutationally invariant 1D Gaussian density
    profile :math:`P(s; \mathbf{R})` along Cartesian axis ``axis`` and
    evaluates a statistical divergence or functional metric
    :math:`\xi(\mathbf{R}) = D(P(\mathbf{R}) \,\|\, Q)` against a target
    reference distribution :math:`Q(s)` (default: uniform distribution
    :math:`Q(s) = 1 / L`).

    Parameters
    ----------
    indices : list[int]
        Indices of atoms included in the density profile.
    axis : int, optional
        Cartesian axis: ``0`` (X), ``1`` (Y), or ``2`` (Z). Default ``2``.
    box_length : float, optional
        Length :math:`L` of the evaluation domain / simulation box along
        ``axis`` (in simulation length units). Default ``1.0``.
    num_bins : int, optional
        Number of evaluation grid nodes :math:`M`. Default ``100``.
    bandwidth : float, optional
        Gaussian kernel standard deviation :math:`\sigma` (length units).
        Default ``0.5``.
    reference : Optional[array_like], optional
        Target reference density profile :math:`Q(s_m)` of shape
        ``(num_bins,)``. If ``None``, defaults to the uniform density
        :math:`1 / L`.
    metric : str, optional
        Divergence / metric type: ``'kl'``, ``'wasserstein'``, ``'sobolev'``,
        ``'js'``, ``'hellinger'``, ``'l2'``, ``'l1'``, or ``'ln'``.
        Default ``'kl'``.
    order : Optional[float], optional
        Order :math:`p` for Wasserstein (default ``1.0``) or :math:`n` for
        :math:`L^n` (default ``2.0``).
    grad_weight : float, optional
        Gradient term weight :math:`\alpha` for ``metric='sobolev'``.
        Default ``1.0``.
    periodic : bool, optional
        If ``True``, uses periodic circular-mean centering, wrapped periodic
        Gaussian kernels, and sets ``requires_box_unwrapping = False``.
        Default ``True``.
    center_com : bool, optional
        If ``True``, centers coordinates on the (periodic or arithmetic)
        center of mass along ``axis`` for translational invariance.
        Default ``True``.
    weights : Optional[array_like], optional
        Optional positive per-atom weights of length ``len(indices)``.
    normalize : bool, optional
        If ``True``, normalizes :math:`P(s)` so :math:`\int P(s) ds = 1`.
        Default ``True``.
    eps : float, optional
        Smoothing parameter for :math:`C^\infty` differentiability.
        Default ``1e-12``.

    Example
    -------
    >>> from pysages.colvars import AxialDensityDivergence
    >>> cv = AxialDensityDivergence(
    ...     indices=list(range(512)),
    ...     axis=2,
    ...     box_length=10.0,
    ...     num_bins=100,
    ...     bandwidth=0.5,
    ...     metric="wasserstein",
    ... )
    """

    def __init__(
        self,
        indices,
        axis: int = 2,
        box_length: float = 1.0,
        num_bins: int = 100,
        bandwidth: float = 0.5,
        reference=None,
        metric: str = "kl",
        order: float = None,
        grad_weight: float = 1.0,
        periodic: bool = True,
        center_com: bool = True,
        weights=None,
        normalize: bool = True,
        eps: float = 1.0e-12,
    ):
        super().__init__(indices, axis)
        if box_length <= 0.0:
            raise ValueError(f"`box_length` must be positive, got {box_length}.")
        if num_bins < 2:
            raise ValueError(f"`num_bins` must be >= 2, got {num_bins}.")
        if bandwidth <= 0.0:
            raise ValueError(f"`bandwidth` must be positive, got {bandwidth}.")
        if metric.lower() not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Valid options: {VALID_METRICS}.")

        self.box_length = float(box_length)
        self.num_bins = int(num_bins)
        self.bandwidth = float(bandwidth)
        self.metric = metric.lower()
        self.order = None if order is None else float(order)
        self.grad_weight = float(grad_weight)
        self.periodic = bool(periodic)
        self.center_com = bool(center_com)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.requires_box_unwrapping = not self.periodic

        if reference is None:
            self.reference = numpy.full(self.num_bins, 1.0 / self.box_length, dtype=float)
        else:
            ref = numpy.asarray(reference, dtype=float)
            if ref.shape != (self.num_bins,):
                raise ValueError(
                    f"`reference` shape {ref.shape} does not match (num_bins,) = ({self.num_bins},)."
                )
            self.reference = ref.copy()

        if weights is None:
            self.weights = None
        else:
            w = numpy.asarray(weights, dtype=float)
            if w.shape != (self.indices.size,):
                raise ValueError(
                    f"`weights` shape {w.shape} does not match number of indices ({self.indices.size},)."
                )
            self.weights = w.copy()

    @property
    def function(self):
        axis = self.axis
        box_length = self.box_length
        num_bins = self.num_bins
        bandwidth = self.bandwidth
        reference = jnp.asarray(self.reference)
        metric = self.metric
        order = self.order
        grad_weight = self.grad_weight
        periodic = self.periodic
        center_com = self.center_com
        weights = None if self.weights is None else jnp.asarray(self.weights)
        normalize = self.normalize
        eps = self.eps
        dx = box_length / num_bins

        def evaluate(positions):
            _, density = axial_density_profile(
                positions,
                axis=axis,
                box_length=box_length,
                num_bins=num_bins,
                bandwidth=bandwidth,
                periodic=periodic,
                center_com=center_com,
                weights=weights,
                normalize=normalize,
            )
            return compute_divergence(
                density,
                reference,
                dx,
                metric=metric,
                order=order,
                grad_weight=grad_weight,
                eps=eps,
            )

        return evaluate


# ============================================================ #
#   2. Coordination / Radial Distribution WAXED CV             #
# ============================================================ #


def coordination_distribution(
    r1,
    r2,
    bins,
    box=None,
    box_inv=None,
    activation: str = "rational",
    n: float = 9.0,
    m: float = 18.0,
    kappa: float = 10.0,
    sigma: float = 0.05,
    normalize: bool = False,
    dx=None,
    eps: float = 1.0e-12,
):
    r"""
    Compute the permutationally invariant pairwise radial / cumulative
    coordination distribution between atom groups ``r1`` and ``r2`` evaluated
    on radial grid ``bins`` :math:`\{s_m\}_{m=1}^M`:

    .. math::

        P(s_m; \mathbf{R}) = \frac{1}{\mathcal{Z}} \sum_{i \in A} \sum_{j \in B}
        \phi(r_{ij}; s_m)

    where :math:`r_{ij} = \sqrt{\|\Delta_{\mathrm{mic}}(\mathbf{r}_i - \mathbf{r}_j)\|_2^2 + \epsilon^2}`.
    """
    pts1 = jnp.atleast_2d(r1)
    pts2 = jnp.atleast_2d(r2)
    dr = pts1[:, None, :] - pts2[None, :, :]
    dr_mic = mic_displacement(dr, box=box, box_inv=box_inv)
    r_sq = jnp.sum(dr_mic * dr_mic, axis=-1)

    act = activation.lower()
    if act == "rational":
        n_f = float(n)
        m_f = float(m)
        n_int = int(round(n_f))
        if abs(m_f - 2.0 * n_f) < 1.0e-12 and abs(n_f - n_int) < 1.0e-12 and n_int % 2 == 0:
            # Avoid sqrt altogether for even n with m = 2n
            u2 = r_sq[:, :, None] / (bins[None, None, :] ** 2)
            phi = 1.0 / (1.0 + jnp.power(u2, n_int // 2))
        else:
            dists = jnp.sqrt(r_sq + eps * eps)
            phi = rational_switching(dists[:, :, None], bins[None, None, :], n=n_f, m=m_f)
    elif act == "sigmoid":
        dists = jnp.sqrt(r_sq + eps * eps)
        phi = sigmoid_switching(dists[:, :, None], bins[None, None, :], kappa=kappa)
    elif act == "gaussian":
        dists = jnp.sqrt(r_sq + eps * eps)
        phi = gaussian_activation(dists[:, :, None], bins[None, None, :], sigma=sigma)
    else:
        raise ValueError(f"Unknown activation '{activation}'. Valid options: {VALID_ACTIVATIONS}.")

    profile = jnp.sum(phi, axis=(0, 1))
    if normalize:
        if dx is None:
            dx = bins[1] - bins[0]
        total = jnp.sum(profile * dx) + 1.0e-24
        profile = profile / total
    return profile


class CoordinationDistributionDivergence(TwoPointCV):
    r"""
    WAXED pairwise radial / cumulative coordination distribution divergence CV.

    Evaluates a permutationally invariant coordination shell distribution
    :math:`P(r_m; \mathbf{R})` between two atom groups (e.g., an ion and
    surrounding solvent oxygens/hydrogens) over radial grid points
    :math:`\{r_m\}_{m=1}^M`, and computes a statistical divergence or
    functional metric :math:`\xi(\mathbf{R}) = D(P(\mathbf{R}) \,\|\, Q)`
    relative to a reference profile :math:`Q(r_m)`.

    Supports general orthorhombic and triclinic periodic cells via the
    Minimum Image Convention (MIC) when ``box`` is provided
    (``requires_box_unwrapping = False``), or unwrapped Euclidean distances
    when ``box=None`` (``requires_box_unwrapping = True``).

    Parameters
    ----------
    indices : list
        Two atom groups ``[group_a, group_b]`` (or two atom indices ``[i, j]``).
    reference : array_like
        Target reference coordination distribution :math:`Q(r_m)` of shape ``(M,)``.
    bins : array_like
        Strictly increasing positive radial grid points :math:`\{r_m\}_{m=1}^M`
        (in simulation length units).
    box : Optional[array_like], optional
        Periodic cell: ``None`` (non-periodic), scalar ``L``, ``(3,)``
        orthorhombic lengths, or ``(3, 3)`` triclinic cell matrix ``H``
        (rows are lattice vectors). Default ``None``.
    metric : str, optional
        Divergence / metric type: ``'kl'``, ``'wasserstein'``, ``'sobolev'``,
        ``'js'``, ``'hellinger'``, ``'l2'``, ``'l1'``, or ``'ln'``.
        Default ``'kl'``.
    activation : str, optional
        Switching / kernel activation: ``'rational'``, ``'sigmoid'``, or
        ``'gaussian'``. Default ``'rational'``.
    n : float, optional
        Numerator exponent for ``activation='rational'``. Default ``9``.
    m : float, optional
        Denominator exponent for ``activation='rational'``. Default ``18``
        (giving :math:`1 / (1 + (r / r_m)^9)`).
    kappa : float, optional
        Steepness parameter for ``activation='sigmoid'`` (inverse length units).
        Default ``10.0``.
    sigma : float, optional
        Gaussian width for ``activation='gaussian'`` (length units).
        Default ``0.05``.
    order : Optional[float], optional
        Exponent :math:`p` for Wasserstein (default ``1.0``) or :math:`n` for
        :math:`L^n` (default ``2.0``).
    grad_weight : float, optional
        Spatial gradient weight for ``metric='sobolev'``. Default ``1.0``.
    normalize : bool, optional
        If ``True``, normalizes :math:`P(r)` to integrate to 1 over ``bins``.
        Default ``False`` (raw cumulative coordination profile).
    eps : float, optional
        Smoothing parameter ensuring :math:`C^\infty` differentiability.
        Default ``1e-12``.

    Example
    -------
    >>> import numpy as np
    >>> from pysages.colvars import CoordinationDistributionDivergence
    >>> bins = np.linspace(0.15, 0.55, 40)
    >>> ref = np.linspace(0.01, 25.0, 40)
    >>> cv = CoordinationDistributionDivergence(
    ...     indices=[[0], list(range(1, 50))],
    ...     reference=ref,
    ...     bins=bins,
    ...     box=[3.0, 3.0, 3.0],
    ...     metric="kl",
    ...     activation="rational",
    ...     n=9,
    ...     m=18,
    ... )
    """

    def __init__(
        self,
        indices,
        reference,
        bins,
        box=None,
        metric: str = "kl",
        activation: str = "rational",
        n: float = 9.0,
        m: float = 18.0,
        kappa: float = 10.0,
        sigma: float = 0.05,
        order: float = None,
        grad_weight: float = 1.0,
        normalize: bool = False,
        eps: float = 1.0e-12,
    ):
        super().__init__(indices)
        bins_arr = numpy.asarray(bins, dtype=float)
        ref_arr = numpy.asarray(reference, dtype=float)
        if bins_arr.ndim != 1 or bins_arr.size < 2:
            raise ValueError("`bins` must be a 1D array with at least 2 radial points.")
        if numpy.any(bins_arr <= 0.0) and activation.lower() == "rational":
            raise ValueError("`bins` must be strictly positive for rational switching.")
        if ref_arr.shape != bins_arr.shape:
            raise ValueError(
                f"`reference` shape {ref_arr.shape} must match `bins` shape {bins_arr.shape}."
            )
        if metric.lower() not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Valid options: {VALID_METRICS}.")
        if activation.lower() not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Valid options: {VALID_ACTIVATIONS}."
            )
        if float(n) <= 0.0 or float(m) <= 0.0:
            raise ValueError("`n` and `m` must be strictly positive.")
        if float(kappa) <= 0.0:
            raise ValueError("`kappa` must be strictly positive.")
        if float(sigma) <= 0.0:
            raise ValueError("`sigma` must be strictly positive.")

        H, H_inv = _parse_box(box)
        self.bins = bins_arr.copy()
        self.dx = _compute_quadrature_dx(self.bins)
        self.reference = ref_arr.copy()
        self.box = H
        self.box_inv = H_inv
        self.metric = metric.lower()
        self.activation = activation.lower()
        self.n = float(n)
        self.m = float(m)
        self.kappa = float(kappa)
        self.sigma = float(sigma)
        self.order = None if order is None else float(order)
        self.grad_weight = float(grad_weight)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.requires_box_unwrapping = self.box is None

    @property
    def function(self):
        bins = jnp.asarray(self.bins)
        dx = self.dx if isinstance(self.dx, float) else jnp.asarray(self.dx)
        grid = None if isinstance(self.dx, float) else bins
        reference = jnp.asarray(self.reference)
        box = None if self.box is None else jnp.asarray(self.box)
        box_inv = None if self.box_inv is None else jnp.asarray(self.box_inv)
        metric = self.metric
        activation = self.activation
        n, m = self.n, self.m
        kappa, sigma = self.kappa, self.sigma
        order = self.order
        grad_weight = self.grad_weight
        normalize = self.normalize
        eps = self.eps

        def evaluate(r1, r2):
            profile = coordination_distribution(
                r1,
                r2,
                bins,
                box=box,
                box_inv=box_inv,
                activation=activation,
                n=n,
                m=m,
                kappa=kappa,
                sigma=sigma,
                normalize=normalize,
                dx=dx,
                eps=eps,
            )
            return compute_divergence(
                profile,
                reference,
                dx,
                metric=metric,
                order=order,
                grad_weight=grad_weight,
                grid=grid,
                eps=eps,
            )

        return evaluate


# ============================================================ #
#   3. Electrostatic Potential Distribution WAXED CV           #
# ============================================================ #


def electrostatic_potential_profile(
    r1,
    r2,
    r3,
    charges,
    bins,
    box=None,
    box_inv=None,
    coulomb_constant: float = COULOMB_KJ_NM_MOL,
    softening: float = 1.0e-12,
):
    r"""
    Compute the electrostatic potential profile contributed by charged atoms
    ``r1`` (charges ``charges``) evaluated along fractional coordinates
    :math:`s_m \in \text{bins}` of the bond vector connecting ``r2`` to ``r3``:

    .. math::

        \mathbf{r}_j(s_m) = \mathbf{r}_{2,j} + s_m \, \Delta_{\mathrm{mic}}(\mathbf{r}_{3,j} - \mathbf{r}_{2,j})

    .. math::

        V(s_m; \mathbf{R}) = k_e \sum_{j=1}^{N_b} \sum_{i=1}^{N_a}
        \frac{q_i}{\sqrt{\|\Delta_{\mathrm{mic}}(\mathbf{r}_{1,i} - \mathbf{r}_j(s_m))\|_2^2 + \epsilon_r^2}}

    When ``r2`` and ``r3`` have unequal group sizes, their respective
    barycenters are used as the bond endpoints.
    """
    pts1 = jnp.atleast_2d(r1)
    pts2 = jnp.atleast_2d(r2)
    pts3 = jnp.atleast_2d(r3)
    if pts2.shape[0] != pts3.shape[0]:
        pts2 = barycenter(pts2)[None, :]
        pts3 = barycenter(pts3)[None, :]

    bond_vec = mic_displacement(pts3 - pts2, box=box, box_inv=box_inv)
    # Probe positions along bond(s): shape (M, N_b, 3)
    probes = pts2[None, :, :] + bins[:, None, None] * bond_vec[None, :, :]

    # Displacements from each probe to each charged atom: shape (M, N_b, N_a, 3)
    dr = pts1[None, None, :, :] - probes[:, :, None, :]
    dr_mic = mic_displacement(dr, box=box, box_inv=box_inv)
    dists = safe_norm(dr_mic, eps=softening)

    # Sum over charged atoms and bond probes -> shape (M,)
    potential = coulomb_constant * jnp.sum(charges[None, None, :] / dists, axis=(1, 2))
    return potential


class ElectrostaticPotentialDivergence(ThreePointCV):
    r"""
    WAXED inverse-distance (:math:`1/r`) electrostatic potential distribution CV.

    Evaluates the electrostatic potential :math:`V(s_m; \mathbf{R})` exerted by
    a set of charged environment/solvent atoms (group 1) at fractional points
    :math:`s_m \in [0, 1]` along the bond connecting atom/group 2 to
    atom/group 3, and measures its functional divergence relative to a target
    reference potential profile :math:`V_{\mathrm{ref}}(s_m)`.

    Because electrostatic potentials can take both positive and negative values,
    signed-safe functional metrics (``'l2'``, ``'ln'``, ``'l1'``, ``'sobolev'``,
    ``'wasserstein'``) operate directly on raw signed profiles without artificial
    clamping.

    Parameters
    ----------
    indices : list
        Three groups/atoms ``[charged_atoms, bond_start, bond_end]``.
    charges : array_like
        Partial atomic charges :math:`q_i` (in elementary charge :math:`e`) for
        each atom in ``charged_atoms``.
    reference : array_like
        Target reference electrostatic potential profile of shape ``(M,)``.
    bins : array_like
        Strictly increasing 1D array of fractional coordinates :math:`s_m`
        (typically in :math:`[0, 1]`) of shape ``(M,)``.
    box : Optional[array_like], optional
        Periodic cell: ``None`` (unwrapped), scalar ``L``, ``(3,)``
        orthorhombic lengths, or ``(3, 3)`` triclinic cell matrix ``H``.
        Default ``None``.
    metric : str, optional
        Functional metric / divergence: ``'l2'`` (default), ``'ln'``, ``'l1'``,
        ``'sobolev'``, ``'wasserstein'``, ``'kl'``, ``'js'``, or ``'hellinger'``.
    coulomb_constant : float, optional
        Electrostatic prefactor :math:`k_e = 1 / (4 \pi \varepsilon_0)`.
        Default ``138.93545764438198`` (:math:`\mathrm{kJ\cdot nm / (mol \cdot e^2)}`).
    softening : float, optional
        Short-range softening length :math:`\epsilon_r` in
        :math:`\sqrt{r^2 + \epsilon_r^2}` preventing division by zero at
        :math:`r = 0`. Default ``1e-12``.
    order : Optional[float], optional
        Exponent for ``'wasserstein'`` (default ``1.0``) or ``'ln'`` (default ``2.0``).
    grad_weight : float, optional
        Gradient weight for ``metric='sobolev'``. Default ``1.0``.
    eps : float, optional
        Smoothing parameter for the metric layer. Default ``1e-12``.

    Example
    -------
    >>> import numpy as np
    >>> from pysages.colvars import ElectrostaticPotentialDivergence
    >>> bins = np.linspace(0.0, 1.0, 20)
    >>> ref = np.linspace(70.0, 150.0, 20)
    >>> charges = np.array([-0.834, 0.417, 0.417])
    >>> cv = ElectrostaticPotentialDivergence(
    ...     indices=[[0, 1, 2], 3, 4],
    ...     charges=charges,
    ...     reference=ref,
    ...     bins=bins,
    ...     box=[3.0, 3.0, 3.0],
    ...     metric="l2",
    ... )
    """

    def __init__(
        self,
        indices,
        charges,
        reference,
        bins,
        box=None,
        metric: str = "l2",
        coulomb_constant: float = COULOMB_KJ_NM_MOL,
        softening: float = 1.0e-12,
        order: float = None,
        grad_weight: float = 1.0,
        eps: float = 1.0e-12,
    ):
        super().__init__(indices)
        charges_arr = numpy.asarray(charges, dtype=float)
        bins_arr = numpy.asarray(bins, dtype=float)
        ref_arr = numpy.asarray(reference, dtype=float)

        n_charged = len(self.groups[0]) if len(self.groups) > 0 else 1
        if charges_arr.ndim != 1 or charges_arr.size != n_charged:
            raise ValueError(
                f"`charges` must be a 1D array of length {n_charged}, got shape {charges_arr.shape}."
            )
        if bins_arr.ndim != 1 or bins_arr.size < 2:
            raise ValueError("`bins` must be a 1D array with at least 2 grid points.")
        if ref_arr.shape != bins_arr.shape:
            raise ValueError(
                f"`reference` shape {ref_arr.shape} must match `bins` shape {bins_arr.shape}."
            )
        if metric.lower() not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Valid options: {VALID_METRICS}.")
        if float(softening) <= 0.0:
            raise ValueError("`softening` must be strictly positive.")

        H, H_inv = _parse_box(box)
        self.charges = charges_arr.copy()
        self.bins = bins_arr.copy()
        self.dx = _compute_quadrature_dx(self.bins)
        self.reference = ref_arr.copy()
        self.box = H
        self.box_inv = H_inv
        self.metric = metric.lower()
        self.coulomb_constant = float(coulomb_constant)
        self.softening = float(softening)
        self.order = None if order is None else float(order)
        self.grad_weight = float(grad_weight)
        self.eps = float(eps)
        self.requires_box_unwrapping = self.box is None

    @property
    def function(self):
        charges = jnp.asarray(self.charges)
        bins = jnp.asarray(self.bins)
        dx = self.dx if isinstance(self.dx, float) else jnp.asarray(self.dx)
        grid = None if isinstance(self.dx, float) else bins
        reference = jnp.asarray(self.reference)
        box = None if self.box is None else jnp.asarray(self.box)
        box_inv = None if self.box_inv is None else jnp.asarray(self.box_inv)
        metric = self.metric
        coulomb_constant = self.coulomb_constant
        softening = self.softening
        order = self.order
        grad_weight = self.grad_weight
        eps = self.eps

        def evaluate(r1, r2, r3):
            profile = electrostatic_potential_profile(
                r1,
                r2,
                r3,
                charges,
                bins,
                box=box,
                box_inv=box_inv,
                coulomb_constant=coulomb_constant,
                softening=softening,
            )
            return compute_divergence(
                profile,
                reference,
                dx,
                metric=metric,
                order=order,
                grad_weight=grad_weight,
                grid=grid,
                eps=eps,
            )

        return evaluate


# ============================================================ #
#   4. WAXED Two-State Classifier / Interpolator CV            #
# ============================================================ #


def _wrap_subcv_fn(cv: CollectiveVariable):
    """Build a single-argument callable evaluating ``cv`` on its local positions."""
    fn = cv.function
    nargs = _get_nargs(fn)
    groups = [g.copy() for g in cv.groups]

    if nargs == 1:
        return fn
    if len(groups) == 0:
        return lambda pos: fn(*pos)
    return lambda pos: fn(*[pos[g] for g in groups])


class DistributionClassifier(CollectiveVariable):
    r"""
    WAXED state-interpolation classifier collective variable.

    Combines two scalar divergence or structural-metric CVs
    :math:`D_A(\mathbf{R})` and :math:`D_B(\mathbf{R})` measuring dissimilarity
    to metastable reference states :math:`A` and :math:`B` into a single smooth
    reaction coordinate:

    - ``mode='ratio'`` (default, range :math:`[0, 1]` for non-negative metrics):

      .. math::

          \xi(\mathbf{R}) = \frac{D_A(\mathbf{R})}{D_A(\mathbf{R}) + D_B(\mathbf{R}) + \epsilon}

    - ``mode='difference'`` (range :math:`[-1, 1]` for non-negative metrics):

      .. math::

          \xi(\mathbf{R}) = \frac{D_A(\mathbf{R}) - D_B(\mathbf{R})}{D_A(\mathbf{R}) + D_B(\mathbf{R}) + \epsilon}

    - ``mode='softmax'`` (range :math:`(0, 1)`, safe for arbitrary signed divergences):

      .. math::

          \xi(\mathbf{R}) = \sigma\!\left(\beta \left[D_A(\mathbf{R}) - D_B(\mathbf{R})\right]\right)

    Parameters
    ----------
    cv_a : CollectiveVariable
        Scalar collective variable measuring divergence to reference state A.
    cv_b : CollectiveVariable
        Scalar collective variable measuring divergence to reference state B.
    mode : str, optional
        Interpolation mode: ``'ratio'`` (default), ``'difference'``, or
        ``'softmax'``.
    beta : float, optional
        Inverse temperature / sharpness for ``mode='softmax'``. Default ``1.0``.
    eps : float, optional
        Denominator regularization for ``'ratio'`` and ``'difference'``.
        Default ``1e-12``.
    """

    def __init__(
        self,
        cv_a: CollectiveVariable,
        cv_b: CollectiveVariable,
        mode: str = "ratio",
        beta: float = 1.0,
        eps: float = 1.0e-12,
    ):
        if cv_a.multicomponent or cv_b.multicomponent:
            raise ValueError("`DistributionClassifier` requires scalar `cv_a` and `cv_b`.")
        if mode.lower() not in VALID_CLASSIFIER_MODES:
            raise ValueError(
                f"Unknown classifier mode '{mode}'. Valid options: {VALID_CLASSIFIER_MODES}."
            )

        idx_a = numpy.asarray(cv_a.indices, dtype=numpy.uint32)
        idx_b = numpy.asarray(cv_b.indices, dtype=numpy.uint32)

        if numpy.array_equal(idx_a, idx_b):
            union_indices = idx_a.copy()
            map_a = numpy.arange(union_indices.size, dtype=numpy.int32)
            map_b = numpy.arange(union_indices.size, dtype=numpy.int32)
        else:
            seen = {}
            union_list = []
            for idx in numpy.concatenate([idx_a, idx_b]):
                val = int(idx)
                if val not in seen:
                    seen[val] = len(union_list)
                    union_list.append(val)
            union_indices = numpy.asarray(union_list, dtype=numpy.uint32)
            map_a = numpy.asarray([seen[int(i)] for i in idx_a], dtype=numpy.int32)
            map_b = numpy.asarray([seen[int(i)] for i in idx_b], dtype=numpy.int32)

        super().__init__(union_indices.tolist())
        self.cv_a = cv_a
        self.cv_b = cv_b
        self._map_a = map_a
        self._map_b = map_b
        self.mode = mode.lower()
        self.beta = float(beta)
        self.eps = float(eps)
        self.requires_box_unwrapping = bool(
            cv_a.requires_box_unwrapping or cv_b.requires_box_unwrapping
        )

    @property
    def function(self):
        eval_a = _wrap_subcv_fn(self.cv_a)
        eval_b = _wrap_subcv_fn(self.cv_b)
        map_a = self._map_a
        map_b = self._map_b
        mode = self.mode
        beta = self.beta
        eps = self.eps

        def evaluate(positions):
            d_a = eval_a(positions[map_a])
            d_b = eval_b(positions[map_b])
            if mode == "ratio":
                return d_a / (d_a + d_b + eps)
            if mode == "difference":
                return (d_a - d_b) / (d_a + d_b + eps)
            return nn.sigmoid(beta * (d_a - d_b))

        return evaluate
