# funnel functions
from functools import partial

import jax.numpy as np
from jax import grad, jit
from jax.numpy import linalg as _linalg

from pysages.colvars.core import build as _build_cvs
from pysages.colvars.funnels import center as _center
from pysages.colvars.funnels import kabsch as _kabsch
from pysages.colvars.funnels import periodic as _periodic
from pysages.colvars.funnels import perp_projection_mobile, projection_mobile


def y_function(x, Z_0, Zcc, R):
    m = (R - Z_0) / Zcc
    return m * x + Z_0


def cone(x, x_p, Zcc, Z_0, R, k):
    F = x_p - y_function(x, Z_0, Zcc, R)
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def cylinder(x, R, k):
    F = x - R
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def borderU(x, k, cv_max):
    B = x - cv_max  # upper limit
    return np.where(B < 0.0, 0.0, 0.5 * k * B * B)


def borderL(x, k, cv_min):
    B = x - cv_min  # lower limit
    return np.where(B > 0.0, 0.0, 0.5 * k * B * B)


def funnel(xi_par, xi_perp, Zcc, Z_0, R, k, k_cv, cv_min, cv_max):
    return np.where(
        xi_par < Zcc,
        cone(xi_par, xi_perp, Zcc, Z_0, R, k)
        + borderL(xi_par, k_cv, cv_min)
        + borderU(xi_par, k_cv, cv_max),
        cylinder(xi_perp, R, k) + borderU(xi_par, k_cv, cv_max) + borderL(xi_par, k_cv, cv_min),
    )


def intermediate_funnel(
    pos,
    ids,
    indexes,
    references,
    weights_ligand,
    weights_protein,
    A,
    B,
    Zcc,
    Z_0,
    R,
    k,
    k_cv,
    cv_min,
    cv_max,
    box,
):
    indices_ligand = np.array(indexes[0])
    indices_protein = np.array(indexes[1])
    indices_anchor = np.array(indexes[2])
    pos_anchor = pos[ids[indices_anchor]]
    pos_protein = pos[ids[indices_protein]]
    pos_ligand = pos[ids[indices_ligand]]
    xi_par = projection_mobile(
        pos_ligand,
        pos_protein,
        pos_anchor,
        np.asarray(references),
        weights_ligand,
        weights_protein,
        np.asarray(A),
        np.asarray(B),
        np.asarray(box),
    )
    xi_perp = perp_projection_mobile(
        pos_ligand,
        pos_protein,
        pos_anchor,
        np.asarray(references),
        weights_ligand,
        weights_protein,
        np.asarray(A),
        np.asarray(B),
        np.asarray(box),
    )
    return funnel(xi_par, xi_perp, Zcc, Z_0, R, k, k_cv, cv_min, cv_max)


def log_funnel(
    pos,
    ids,
    indexes,
    references,
    weights_ligand,
    weights_protein,
    A,
    B,
    Zcc,
    Z_0,
    R,
    k,
    k_cv,
    cv_min,
    cv_max,
    box,
):
    indices_ligand = np.array(indexes[0])
    indices_protein = np.array(indexes[1])
    indices_anchor = np.array(indexes[2])
    pos_anchor = pos[ids[indices_anchor]]
    pos_protein = pos[ids[indices_protein]]
    pos_ligand = pos[ids[indices_ligand]]
    xi_perp = perp_projection_mobile(
        pos_ligand,
        pos_protein,
        pos_anchor,
        np.asarray(references),
        weights_ligand,
        weights_protein,
        np.asarray(A),
        np.asarray(B),
        np.asarray(box),
    )
    return xi_perp


def external_funnel(
    data,
    indexes,
    references,
    weights_ligand,
    weights_protein,
    A,
    B,
    Zcc,
    Z_0,
    R,
    k,
    k_cv,
    cv_min,
    cv_max,
    box,
):
    pos = data.positions[:, :3]
    ids = data.indices
    bias = grad(intermediate_funnel)(
        pos,
        ids,
        indexes,
        references,
        weights_ligand,
        weights_protein,
        A,
        B,
        Zcc,
        Z_0,
        R,
        k,
        k_cv,
        cv_min,
        cv_max,
        box,
    )
    proj = log_funnel(
        pos,
        ids,
        indexes,
        references,
        weights_ligand,
        weights_protein,
        A,
        B,
        Zcc,
        Z_0,
        R,
        k,
        k_cv,
        cv_min,
        cv_max,
        box,
    )
    return bias, proj


def get_funnel_force(
    indices_sys,
    ref_positions,
    A,
    B,
    Zcc,
    Z_0,
    R_cyl,
    k_cone,
    k_cv,
    cv_min,
    cv_max,
    cv_buffer,
    box,
    w_ligand=None,
    w_protein=None,
):
    funnel_force = partial(
        external_funnel,
        indexes=indices_sys,
        references=ref_positions,
        weights_ligand=w_ligand,
        weights_protein=w_protein,
        A=A,
        B=B,
        Zcc=Zcc,
        Z_0=Z_0,
        R=R_cyl,
        k=k_cone,
        k_cv=k_cv,
        cv_min=cv_min - cv_buffer,
        cv_max=cv_max + cv_buffer,
        box=box,
    )
    return jit(funnel_force)


# =============================================================================
# KEEP-OUT CAPSULE.  A restrained group (e.g. a counter-ion) may not enter a
# capsule of radius R about the segment [A + zlo*n, A + zhi*n] of the mobile host
# axis (n = (B-A)/|B-A|, A/B in the host reference frame, hemispherical ends):
#     E = 0.5 k (R - d)^2  for d < R,  else 0,
# d = distance from the group's centroid to the segment.  Same mobile-axis
# machinery as the funnel (Kabsch fit of the host onto its reference), so the
# JAX gradient carries the reaction forces onto the host atoms.
# PBC: the minimum image of the group is taken about the CAPSULE CENTRE mapped
# into the simulation frame.  The energy is then continuous under PBC iff the
# capsule fits inside the half-box sphere, R + (zhi - zlo)/2 < L_min/2 (any
# point on the minimum-image boundary is then outside the capsule for both
# images).  An infinite cylinder about a tumbling axis is NOT well defined
# under PBC (its images come arbitrarily close to every point); hence the caps.
# =============================================================================


def capsule_keepout(d, R, k):
    F = R - d
    return np.where(F < 0.0, 0.0, 0.5 * k * F * F)


def keepout_distance(
    pos, ids, indexes, references, weights_group, weights_protein, A, B, zlo, zhi, box
):
    """Distance (host reference frame) from the minimum-image centroid of the group
    indexes[0] to the axis segment [A + zlo*n, A + zhi*n]; indexes[1] = host atoms."""
    indices_group = np.array(indexes[0])
    indices_protein = np.array(indexes[1])
    pos_group = pos[ids[indices_group]]
    pos_protein = pos[ids[indices_protein]]
    references = np.asarray(references)
    A = np.asarray(A)
    B = np.asarray(B)
    box = np.asarray(box)
    com_prot = _center(pos_protein, weights_protein)
    com_ref = _center(references, weights_protein)
    U = _kabsch(pos_protein, references, weights_protein)  # row-vector map: sim -> reference frame
    n = B - A
    n = n / _linalg.norm(n)
    M = A + 0.5 * (zlo + zhi) * n  # capsule centre (reference frame)
    h = 0.5 * (zhi - zlo)
    M_sim = np.dot(M - com_ref, U.T) + com_prot  # capsule centre in the simulation frame
    d_sim = _center(_periodic(pos_group - M_sim, box), weights_group)  # min. image about the centre
    y = np.dot(d_sim, U)  # displacement from M, reference frame
    t = np.clip(np.dot(y, n), -h, h)
    return _linalg.norm(y - t * n)


def intermediate_keepout(
    pos, ids, indexes, references, weights_group, weights_protein, A, B, zlo, zhi, R, k, box
):
    d = keepout_distance(
        pos, ids, indexes, references, weights_group, weights_protein, A, B, zlo, zhi, box
    )
    return capsule_keepout(d, R, k)


def external_keepout(
    data, indexes, references, weights_group, weights_protein, A, B, zlo, zhi, R, k, box
):
    pos = data.positions[:, :3]
    ids = data.indices
    bias = grad(intermediate_keepout)(
        pos, ids, indexes, references, weights_group, weights_protein, A, B, zlo, zhi, R, k, box
    )
    d = keepout_distance(
        pos, ids, indexes, references, weights_group, weights_protein, A, B, zlo, zhi, box
    )
    return bias, d


def external_funnel_keepout(data, funnel_kwargs, keepout_kwargs_list):
    """Funnel on the ligand + one keep-out capsule per restrained group.
    Returns (bias, perp) with perp = the funnel ligand's perpendicular distance
    (the logged quantity), exactly as external_funnel does."""
    bias, proj = external_funnel(data, **funnel_kwargs)
    for kw in keepout_kwargs_list:
        b, _ = external_keepout(data, **kw)
        bias = bias + b
    return bias, proj


def get_funnel_keepout_force(
    indices_sys,
    ref_positions,
    A,
    B,
    Zcc,
    Z_0,
    R_cyl,
    k_cone,
    k_cv,
    cv_min,
    cv_max,
    cv_buffer,
    box,
    keepouts,
    w_ligand=None,
    w_protein=None,
):
    """get_funnel_force plus keep-out capsules.  `keepouts` = list of dicts with keys
    indices (atom indices of the restrained group), zlo, zhi, R, k [, weights]."""
    funnel_kwargs = dict(
        indexes=indices_sys,
        references=ref_positions,
        weights_ligand=w_ligand,
        weights_protein=w_protein,
        A=A,
        B=B,
        Zcc=Zcc,
        Z_0=Z_0,
        R=R_cyl,
        k=k_cone,
        k_cv=k_cv,
        cv_min=cv_min - cv_buffer,
        cv_max=cv_max + cv_buffer,
        box=box,
    )
    half_box = 0.5 * float(min(box))
    kos = []
    for ko in keepouts:
        if ko["R"] + 0.5 * (ko["zhi"] - ko["zlo"]) >= half_box:
            raise ValueError(
                f"keep-out capsule R={ko['R']} z=[{ko['zlo']},{ko['zhi']}] does not fit inside "
                f"the half-box sphere ({half_box:.4f} nm): not continuous under PBC"
            )
        kos.append(
            dict(
                indexes=[list(ko["indices"]), indices_sys[1], indices_sys[2]],
                references=ref_positions,
                weights_group=ko.get("weights"),
                weights_protein=w_protein,
                A=A,
                B=B,
                zlo=ko["zlo"],
                zhi=ko["zhi"],
                R=ko["R"],
                k=ko["k"],
                box=box,
            )
        )
    return jit(
        partial(external_funnel_keepout, funnel_kwargs=funnel_kwargs, keepout_kwargs_list=kos)
    )


# =============================================================================
# GATED WALL.  A harmonic wall on one collective variable that is only active
# while a second collective variable (the gate) lies inside a window:
#     E = s(x_gate) * 0.5 k * [max(0, wall_lower - x_wall)^2 + max(0, x_wall - wall_upper)^2]
# s = 1 for lower_on <= x_gate <= upper_on, with cosine ramps down to 0 at
# lower_off and upper_off, and 0 outside (E and its gradient are continuous).
# Typical use: keep a ligand aligned with the host axis only while it crosses
# the entrance of the binding site, so that both end states stay unrestrained.
# The free energy difference between two states that lie outside the window is
# then not changed by the restraint, as long as the restraint is treated as part
# of the Hamiltonian (its mean force along the gate CV must not be removed).
# =============================================================================


def window_switch(x, lower_off, lower_on, upper_on, upper_off):
    """1 for lower_on <= x <= upper_on, cosine ramps to 0 at lower_off and upper_off."""
    up = np.clip((x - lower_off) / (lower_on - lower_off), 0.0, 1.0)
    down = np.clip((upper_off - x) / (upper_off - upper_on), 0.0, 1.0)
    return 0.25 * (1.0 - np.cos(np.pi * up)) * (1.0 - np.cos(np.pi * down))


def gated_wall_energy(
    x_gate, x_wall, lower_off, lower_on, upper_on, upper_off, wall_lower, wall_upper, k
):
    below = np.maximum(wall_lower - x_wall, 0.0)
    above = np.maximum(x_wall - wall_upper, 0.0)
    s = window_switch(x_gate, lower_off, lower_on, upper_on, upper_off)
    return s * 0.5 * k * (below * below + above * above)


def intermediate_gated_wall(positions, data, gate_cv, wall_cv, **params):
    cvs = _build_cvs(gate_cv, wall_cv, differentiate=False)
    xi = cvs(data._replace(positions=positions)).flatten()
    return gated_wall_energy(xi[0], xi[1], **params)


def external_gated_wall(data, gate_cv, wall_cv, **params):
    """(dE/dpositions, [x_gate, x_wall, s, E]) for one gated wall; same contract as
    external_funnel (the method subtracts the returned gradient from the forces)."""
    bias = grad(intermediate_gated_wall)(data.positions, data, gate_cv, wall_cv, **params)
    cvs = _build_cvs(gate_cv, wall_cv, differentiate=False)
    xi = cvs(data).flatten()
    s = window_switch(
        xi[0], params["lower_off"], params["lower_on"], params["upper_on"], params["upper_off"]
    )
    energy = gated_wall_energy(xi[0], xi[1], **params)
    return bias[:, :3], np.array([xi[0], xi[1], s, energy])


def get_gated_wall_force(
    gate_cv,
    wall_cv,
    lower_off,
    lower_on,
    upper_on,
    upper_off,
    k,
    wall_lower=None,
    wall_upper=None,
):
    """Gated wall on `wall_cv`, switched by `gate_cv` (both single-valued collective
    variables). The window bounds must satisfy lower_off < lower_on <= upper_on <
    upper_off; give `wall_lower` and/or `wall_upper` for the side(s) to restrain."""
    if not lower_off < lower_on <= upper_on < upper_off:
        raise ValueError("The gate needs lower_off < lower_on <= upper_on < upper_off.")
    if wall_lower is None and wall_upper is None:
        raise ValueError("Give `wall_lower` and/or `wall_upper`.")
    return jit(
        partial(
            external_gated_wall,
            gate_cv=gate_cv,
            wall_cv=wall_cv,
            lower_off=lower_off,
            lower_on=lower_on,
            upper_on=upper_on,
            upper_off=upper_off,
            wall_lower=-np.inf if wall_lower is None else wall_lower,
            wall_upper=np.inf if wall_upper is None else wall_upper,
            k=k,
        )
    )
