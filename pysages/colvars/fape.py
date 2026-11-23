# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
SE(3)-invariant Frame Aligned Point Error (FAPE) collective variable.

Repurposes AlphaFold's Frame Aligned Point Error metric as a smooth,
JAX-differentiable collective variable for biomolecular conformational
sampling and catalytic reaction pathway discovery.
"""

import numpy
from jax import nn, vmap
from jax import numpy as jnp

from pysages.colvars.core import CollectiveVariable
from pysages.colvars.distributions import _parse_box, mic_displacement, safe_norm


def construct_frame(positions, triplet, box=None, box_inv=None, eps: float = 1.0e-12):
    r"""
    Construct a right-handed local rigid frame :math:`T = (R, \mathbf{t})` in
    :math:`\mathrm{SE}(3)` from an ordered triplet of local atom indices
    ``(ia, ib, ic)`` via stabilized Gram--Schmidt orthogonalization:

    .. math::

        \mathbf{t} = \mathbf{x}_{i_a}, \quad
        \mathbf{e}_1 = \frac{\Delta\mathbf{x}_{ba}}{\|\Delta\mathbf{x}_{ba}\|_\epsilon}, \quad
        \mathbf{v}_2 = \Delta\mathbf{x}_{ca} - (\Delta\mathbf{x}_{ca} \cdot \mathbf{e}_1)\mathbf{e}_1, \quad
        \mathbf{e}_2 = \frac{\mathbf{v}_2}{\|\mathbf{v}_2\|_\epsilon}, \quad
        \mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2

    Returns
    -------
    R : jax.Array
        Proper rotation matrix of shape ``(3, 3)`` with columns
        ``[e1, e2, e3]``.
    t : jax.Array
        Frame origin vector of shape ``(3,)``.
    """
    ia, ib, ic = triplet[0], triplet[1], triplet[2]
    t = positions[ia]

    v1 = mic_displacement(positions[ib] - t, box=box, box_inv=box_inv)
    e1 = v1 / safe_norm(v1, eps=eps)

    v2 = mic_displacement(positions[ic] - t, box=box, box_inv=box_inv)
    v2 = v2 - jnp.dot(v2, e1) * e1
    e2 = v2 / safe_norm(v2, eps=eps)

    e3 = jnp.cross(e1, e2)
    R = jnp.stack([e1, e2, e3], axis=-1)
    return R, t


def _local_coordinates(positions, R, t, box=None, box_inv=None):
    """Transform atomic positions into local frame ``T = (R, t)``: ``R^T (x - t)``."""
    dr = mic_displacement(positions - t[None, :], box=box, box_inv=box_inv)
    return dr @ R


def _smooth_clamp(d, d_clamp: float, tau: float):
    r"""
    :math:`C^\infty` smooth upper-capping approximation to :math:`\min(d, d_{\mathrm{clamp}})`:

    .. math::

        g_\tau(d) = d - \tau\,\mathrm{softplus}\!\left(\frac{d - d_{\mathrm{clamp}}}{\tau}\right)
                    + \tau\,\mathrm{softplus}\!\left(-\frac{d_{\mathrm{clamp}}}{\tau}\right)

    Satisfies :math:`g_\tau(0) = 0`, :math:`0 < g_\tau'(d) < 1`, and converges
    exponentially to :math:`\min(d, d_{\mathrm{clamp}})` away from
    :math:`d = d_{\mathrm{clamp}}`.
    """
    return (
        d
        - tau * nn.softplus((d - d_clamp) / tau)
        + tau * nn.softplus(-d_clamp / tau)
    )


def fape(
    positions,
    reference,
    frames,
    frame_weights=None,
    d_clamp: float = None,
    length_scale: float = 1.0,
    clamp_smoothness: float = None,
    box=None,
    box_inv=None,
    eps: float = 1.0e-12,
):
    r"""
    Compute the :math:`\mathrm{SE}(3)`-invariant Frame Aligned Point Error
    (FAPE) between ``positions`` and ``reference``:

    .. math::

        \xi_{\mathrm{FAPE}}(\mathbf{R}; \tilde{\mathbf{R}})
        = \frac{1}{Z_{\mathrm{FAPE}} N_{\mathrm{frames}}}
          \sum_{k=1}^{N_{\mathrm{frames}}} \sum_{i=1}^{N_{\mathrm{atoms}}}
          w_{k,i} \, \tilde{d}\!\left(
              \left\| T_k^{-1} \circ \mathbf{x}_i - \tilde{T}_k^{-1} \circ \tilde{\mathbf{x}}_i \right\|_2
          \right)

    Parameters
    ----------
    positions : jax.Array
        Instantaneous Cartesian coordinates of shape ``(N_atoms, 3)``.
    reference : jax.Array
        Reference Cartesian coordinates of shape ``(N_atoms, 3)``.
    frames : jax.Array
        Integer array of local index triplets of shape ``(N_frames, 3)``.
    frame_weights : Optional[jax.Array]
        Per-frame normalized atom weights of shape ``(N_frames, N_atoms)`` with
        each row summing to 1. If ``None``, uniform weights ``1 / N_atoms`` are
        used.
    d_clamp : Optional[float]
        Optional upper clamp distance capping long-range displacements.
    length_scale : float
        Length-scale normalization constant :math:`Z_{\mathrm{FAPE}}`.
    clamp_smoothness : Optional[float]
        Transition width :math:`\tau` for smooth clamping (default ``0.02 * d_clamp``).
    box : Optional[jax.Array]
        Optional periodic simulation cell matrix ``H`` of shape ``(3, 3)``.
    box_inv : Optional[jax.Array]
        Optional inverse cell matrix ``H^{-1}`` of shape ``(3, 3)``.
    eps : float
        Smoothing parameter in Euclidean norms preventing NaN gradients at
        zero displacement or colinear frame triplets.

    Returns
    -------
    jax.Array
        Scalar FAPE collective variable value.
    """
    n_atoms = positions.shape[0]
    if frame_weights is None:
        weights = jnp.full((frames.shape[0], n_atoms), 1.0 / n_atoms)
    else:
        weights = frame_weights

    tau = None
    if d_clamp is not None:
        tau = 0.02 * d_clamp if clamp_smoothness is None else float(clamp_smoothness)

    def frame_error(triplet, w_row):
        R_ref, t_ref = construct_frame(reference, triplet, box=box, box_inv=box_inv, eps=eps)
        R_mod, t_mod = construct_frame(positions, triplet, box=box, box_inv=box_inv, eps=eps)

        local_ref = _local_coordinates(reference, R_ref, t_ref, box=box, box_inv=box_inv)
        local_mod = _local_coordinates(positions, R_mod, t_mod, box=box, box_inv=box_inv)

        dloc = local_mod - local_ref
        per_atom = safe_norm(dloc, eps=eps)
        if d_clamp is not None:
            per_atom = _smooth_clamp(per_atom, d_clamp, tau)
        return jnp.sum(w_row * per_atom)

    frame_vals = vmap(frame_error)(frames, weights)
    return jnp.mean(frame_vals) / length_scale


class FAPE(CollectiveVariable):
    r"""
    :math:`\mathrm{SE}(3)`-invariant Frame Aligned Point Error (FAPE)
    collective variable.

    Measures local-frame-aligned structural dissimilarity between instantaneous
    atomic coordinates :math:`\mathbf{R}` and a target reference configuration
    :math:`\tilde{\mathbf{R}}`:

    .. math::

        \xi_{\mathrm{FAPE}}(\mathbf{R}; \tilde{\mathbf{R}})
        = \frac{1}{Z_{\mathrm{FAPE}} N_{\mathrm{frames}}}
          \sum_{k=1}^{N_{\mathrm{frames}}} \sum_{i=1}^{N_{\mathrm{atoms}}}
          w_{k,i} \, \tilde{d}\!\left(
              \left\| T_k^{-1} \circ \mathbf{x}_i - \tilde{T}_k^{-1} \circ \tilde{\mathbf{x}}_i \right\|_2
          \right)

    where each local rigid frame :math:`T_k = (R_k, \mathbf{t}_k)` is defined
    by an ordered triplet of atoms ``(ia, ib, ic)`` in ``frames`` (specified as
    local indices into ``indices``).

    Exact invariance holds under global 3D translations and proper rotations
    :math:`\mathrm{SE}(3)` (when ``box=None``), while retaining sensitivity to
    chirality, local orientation, and stereochemical rearrangements.

    Parameters
    ----------
    indices : list[int]
        Global atom indices of the selected structure (length ``N_atoms``).
    reference : array_like
        Reference Cartesian coordinates of shape ``(N_atoms, 3)`` matching the
        order of ``indices``.
    frames : Sequence[Tuple[int, int, int]]
        Sequence of ``N_frames`` triplets ``(ia, ib, ic)`` of *local* indices
        in ``0 .. N_atoms - 1`` defining each local Gram--Schmidt frame.
    weights : Optional[array_like], optional
        Optional non-negative per-atom weights of length ``N_atoms``. If
        ``None``, all atoms are weighted equally.
    mask_anchors : bool, optional
        If ``True``, zeros out the weight of each frame's 3 anchor atoms
        ``(ia, ib, ic)`` when scoring that frame (scaffold-anchored FAPE
        response) and renormalizes the remaining atom weights to sum to 1.
        Default ``False`` (standard AlphaFold FAPE).
    d_clamp : Optional[float], optional
        Optional upper clamp distance capping long-range displacements via a
        :math:`C^\infty` smooth clamp. Default ``None`` (unclamped).
    length_scale : float, optional
        Normalization length scale :math:`Z_{\mathrm{FAPE}} > 0`.
        Default ``1.0``.
    clamp_smoothness : Optional[float], optional
        Smoothness width :math:`\tau > 0` for the smooth upper clamp when
        ``d_clamp`` is set. Defaults to ``0.02 * d_clamp``.
    box : Optional[array_like], optional
        Optional periodic cell (scalar, ``(3,)``, or ``(3, 3)``) for MIC
        relative displacements (sets ``requires_box_unwrapping = False``).
        Default ``None`` (unwrapped coordinates, exact :math:`\mathrm{SE}(3)`
        invariance, ``requires_box_unwrapping = True``).
    eps : float, optional
        Smoothing parameter in Euclidean norms ensuring NaN-free first and
        second derivatives. Default ``1e-12``.

    Example
    -------
    >>> import numpy as np
    >>> from pysages.colvars import FAPE
    >>> ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> cv = FAPE(indices=[0, 1, 2, 3], reference=ref, frames=[(0, 1, 2)], d_clamp=1.0)
    """

    def __init__(
        self,
        indices,
        reference,
        frames,
        weights=None,
        mask_anchors: bool = False,
        d_clamp: float = None,
        length_scale: float = 1.0,
        clamp_smoothness: float = None,
        box=None,
        eps: float = 1.0e-12,
    ):
        super().__init__(indices)

        ref_arr = numpy.asarray(reference, dtype=float)
        n_atoms = int(self.indices.size)
        if ref_arr.shape != (n_atoms, 3):
            raise ValueError(
                f"`reference` must have shape ({n_atoms}, 3), got {ref_arr.shape}."
            )

        frames_arr = numpy.asarray(frames, dtype=numpy.int32)
        if frames_arr.ndim != 2 or frames_arr.shape[1] != 3 or frames_arr.shape[0] < 1:
            raise ValueError(
                f"`frames` must be a non-empty sequence of index triplets, got shape {frames_arr.shape}."
            )
        if numpy.any(frames_arr < 0) or numpy.any(frames_arr >= n_atoms):
            raise ValueError(f"Frame indices must be in [0, {n_atoms - 1}].")
        for k, (ia, ib, ic) in enumerate(frames_arr):
            if ia == ib or ia == ic or ib == ic:
                raise ValueError(f"Frame {k} has duplicate anchor indices: {(int(ia), int(ib), int(ic))}.")

        if weights is None:
            base_w = numpy.ones(n_atoms, dtype=float)
        else:
            base_w = numpy.asarray(weights, dtype=float)
            if base_w.shape != (n_atoms,):
                raise ValueError(
                    f"`weights` must have shape ({n_atoms},), got {base_w.shape}."
                )
            if numpy.any(base_w < 0.0):
                raise ValueError("`weights` must be non-negative.")

        # Precompute per-frame normalized weights in NumPy (pickle-safe)
        frame_weights = numpy.broadcast_to(base_w[None, :], (frames_arr.shape[0], n_atoms)).copy()
        if mask_anchors:
            for k, (ia, ib, ic) in enumerate(frames_arr):
                frame_weights[k, [ia, ib, ic]] = 0.0

        row_sums = numpy.sum(frame_weights, axis=1, keepdims=True)
        if numpy.any(row_sums <= 0.0):
            raise ValueError(
                "Every frame must have positive total weight after optional anchor masking."
            )
        frame_weights = frame_weights / row_sums

        if float(length_scale) <= 0.0:
            raise ValueError(f"`length_scale` must be strictly positive, got {length_scale}.")
        if d_clamp is not None and float(d_clamp) <= 0.0:
            raise ValueError(f"`d_clamp` must be strictly positive, got {d_clamp}.")
        if clamp_smoothness is not None and float(clamp_smoothness) <= 0.0:
            raise ValueError(f"`clamp_smoothness` must be strictly positive, got {clamp_smoothness}.")
        if float(eps) <= 0.0:
            raise ValueError(f"`eps` must be strictly positive, got {eps}.")

        H, H_inv = _parse_box(box)
        self.reference = ref_arr.copy()
        self.frames = frames_arr.copy()
        self.weights = base_w.copy()
        self.frame_weights = frame_weights
        self.mask_anchors = bool(mask_anchors)
        self.d_clamp = None if d_clamp is None else float(d_clamp)
        self.length_scale = float(length_scale)
        self.clamp_smoothness = None if clamp_smoothness is None else float(clamp_smoothness)
        self.box = H
        self.box_inv = H_inv
        self.eps = float(eps)
        self.requires_box_unwrapping = self.box is None

    @property
    def function(self):
        ref = jnp.asarray(self.reference)
        frames = jnp.asarray(self.frames)
        frame_weights = jnp.asarray(self.frame_weights)
        d_clamp = self.d_clamp
        length_scale = self.length_scale
        clamp_smoothness = self.clamp_smoothness
        box = None if self.box is None else jnp.asarray(self.box)
        box_inv = None if self.box_inv is None else jnp.asarray(self.box_inv)
        eps = self.eps

        def evaluate(positions):
            return fape(
                positions,
                ref,
                frames,
                frame_weights=frame_weights,
                d_clamp=d_clamp,
                length_scale=length_scale,
                clamp_smoothness=clamp_smoothness,
                box=box,
                box_inv=box_inv,
                eps=eps,
            )

        return evaluate
