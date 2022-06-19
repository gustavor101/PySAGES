# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES


import jax.numpy as np
from jax.numpy import linalg
from .core import ThreePointCV
from pysages.colvars.coordinates import barycenter


class Projection_Parallel(ThreePointCV):
    def __init__(self,indices):
        super().__init__(indices)
        self.requires_box_unwrapping = True

    @property
    def function(self):
        return parallel


def parallel(p1, p2, p3):
    """
    Returns the parallel projection of a point (p3) defined by the axis of two points (pi and p2) in space.
    """
    r1  = barycenter(p1)
    r2  = barycenter(p2)
    r3  = barycenter(p3)
    a = r3 - r1
    b = r2 - r1

    return np.dot(a,b)/linalg.norm(b)

class Projection_Perpendicular(ThreePointCV):
    def __init__(self,indices):
        super().__init__(indices)
        self.requires_box_unwrapping = True
    @property
    def function(self):
        return perpendicular


def perpendicular(p1, p2, p3):
    """
    Returns the perpendicular projection of a point (p3) defined by the axis of two points (pi and p2) in space.
    """
    r1  = barycenter(p1)
    r2  = barycenter(p2)
    r3  = barycenter(p3)
    a = r3 - r1
    b = r2 - r1
    return np.sqrt(np.dot(a,a)-np.dot(a,b)**2/np.dot(b,b))
