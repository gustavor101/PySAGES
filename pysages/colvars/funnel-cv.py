# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective variables for orientiations describe the orientation measures
of particles in the simulation respect to a reference.

It is common to describe such orientations using RMSD, tilt, sping angle
and more to spot inside a molecule or protein a conformation change.
"""

from jax import numpy as np
from jax import vmap
from jax.numpy import linalg

from pysages.colvars.coordinates import barycenter, weighted_barycenter
from pysages.colvars.core import AxisCV, CollectiveVariable, TwoPointCV

_QUATERNION_BASES = {
    0: (0, 1, 0, 0),
    1: (0, 0, 1, 0),
    2: (0, 0, 0, 1),
}


def fitted_positions(positions, references, weights):
    if weights is None:
        pos_b = barycenter(positions)
        ref_b = barycenter(references)
        fit_pos = np.add(positions, -pos_b)
        fit_ref = np.add(references, -ref_b)
    else:
        pos_b = weighted_barycenter(positions, weights)
        ref_b = weighted_barycenter(references, weights)
        fit_pos = np.add(np.multiply(positions, weights), -pos_b)
        fit_ref = np.add(np.multiply(references, weights), -ref_b)
    return fit_pos, fit_ref


def outer(a, b):
    return np.outer(a, b)


def quaternion_matrix(positions, references, weights):
    """
    Function to construct the quaternion matrix based on the reference positions.

    Parameters
    ----------
    positions: np.array
       atomic positions via indices.
    references: np.array
       Cartesian coordinates of the reference positions of the atoms in indices.
       The number of coordinates must match the ones of the atoms used in indices.
    weights: np.array
       Weights for the barycenter calculation
    """
    if len(positions) != len(references):
        raise RuntimeError("References must be of the same length as the positions")
    fit_pos, fit_ref = fitted_positions(positions, references, weights)
    R = vmap(outer, in_axes=(0, 0))(fit_pos, fit_ref).sum(axis=0)
    S_00 = R[0, 0] + R[1, 1] + R[2, 2]
    S_01 = R[1, 2] - R[2, 1]
    S_02 = R[2, 0] - R[0, 2]
    S_03 = R[0, 1] - R[1, 0]
    S_11 = R[0, 0] - R[1, 1] - R[2, 2]
    S_12 = R[0, 1] + R[1, 0]
    S_13 = R[0, 2] + R[2, 0]
    S_22 = -R[0, 0] + R[1, 1] - R[2, 2]
    S_23 = R[1, 2] + R[2, 1]
    S_33 = -R[0, 0] + R[1, 1] + R[2, 2]
    S = np.array(
        [
            [S_00, S_01, S_02, S_03],
            [S_01, S_11, S_12, S_13],
            [S_02, S_12, S_22, S_23],
            [S_03, S_13, S_23, S_33],
        ]
    )
    return S


def dist_sq(d):
    return np.dot(d, d)


class Projecton_on_Axis_mobile(TwoPointCV):
    """
    Use a reference to calculate the RMSD of a set of atoms.
    The algorithm is based on https://doi.org/10.1002/jcc.20110.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, references, weights_lig=None, weights_prot=None, A, B):
        super().__init__(indices, 2)
        self.references = np.asarray(references)
        self.weights_lig = np.asarray(weights_lig) if weights_lig else None
        self.weights_prot = np.asarray(weights_prot) if weights_prot else None
        self.A = np.asarray(A)
        self.B = np.asarray(B)

    @property
    def function(self):
        return projection_mobile(
            r1, r2, self.references, self.weights_lig, self.weights_prot, self.A, self.B
        )


def center(positions, weights):
    if weights is None:
        return barycenter(positions)
    else:
        return weighted_barycenter(positions, weights)


def create_matrot(wxyz):
    # https://github.com/brentyi/jaxlie/blob/master/jaxlie/_so3.py
    norm2 = wxyz @ wxyz
    q = wxyz * np.sqrt(2.0 / norm2)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )


def projection_mobile(ligand, backbone, references, weights_lig, weights_prot, A, B):
    com_lig = center(ligand, weights_lig)
    com_prot = center(backbone, weights_prot)
    _, rotmat = linalg.eigh(quaternion_matrix(backbone, references, weights_prot))
    quat = rotmat[:3]
    A_rot = np.matmul(create_matrot(quat), A - com_prot)
    B_rot = np.matmul(create_matrot(quat), B - com_prot)
    vector = B_rot - A_rot
    norm2 = vector @ vector
    eje = vector * np.sqrt(1.0 / norm2)

    return np.dot(eje, com_lig - (A_rot + com_prot))
