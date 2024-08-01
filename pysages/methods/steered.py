# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Steered Molecular Dynamics.

Biasing a simulation towards a value of a collective variable using a time dependent Harmonic Bias.
This method implements such a bias.

The Hamiltonian is amended with a term
:math:`\\mathcal{H} = \\mathcal{H}_0 + \\mathcal{H}_\\mathrm{HB}(\\xi(t))` where
:math:`\\mathcal{H}_\\mathrm{HB}(\\xi) = \\boldsymbol{K}/2 (\\xi_0(t) - \\xi)^2`
biases the simulations around the collective variable :math:`\\xi_0(t)`.
"""

from typing import NamedTuple

from jax import numpy as np

from pysages.methods.bias import Bias
from pysages.methods.core import generalize
#from pysages.utils import JaxArray
from pysages.typing import JaxArray

class SteeredState(NamedTuple):
    """
    Description of a state biased by a harmonic potential for a CV.

    xi: JaxArray
        Collective variable value of the last simulation step.

    bias: JaxArray
        Array with harmonic biasing forces for each particle in the simulation.

    centers: JaxArray
        Moving centers of the harmonic bias applied.

    forces: JaxArray
        Array with harmonic forces for each collective variable in the simulation.

    work: JaxArray
        Array with the current work applied in the simulation.
    """

    xi: JaxArray
    bias: JaxArray
    centers: JaxArray
    forces: JaxArray
    work: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Steered(Bias):
    """
    Steered method class.
    """

    __special_args__ = Bias.__special_args__.union({"kspring", "steer_velocity"})

    def __init__(self, cvs, kspring, center, steer_velocity, **kwargs):
        """
        Arguments
        ---------
        cvs: Union[List, Tuple]
            A list or tuple of collective variables, length `N`.
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        center:
            An array of length `N` representing the initial state of the harmonic biasing potential.
        steer_velocity:
            An array of length `N` representing the constant steer_velocity. Units are cvs/time.
        """
        super().__init__(cvs, center, **kwargs)
        self.cv_dimension = len(cvs)
        self.kspring = kspring
        self.steer_velocity = steer_velocity

    def __getstate__(self):
        state, kwargs = super().__getstate__()
        state["kspring"] = self._kspring
        state["steer_velocity"] = self._steer_velocity
        return state, kwargs

    @property
    def kspring(self):
        """
        Retrieve the spring constant.
        """
        return self._kspring

    @kspring.setter
    def kspring(self, kspring):
        """
        Set new spring constant.

        Arguments
        ---------
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        """
        # Ensure array
        kspring = np.asarray(kspring)
        shape = kspring.shape
        N = self.cv_dimension

        if len(shape) > 2:
            raise RuntimeError(f"Wrong kspring shape {shape} (expected scalar, 1D or 2D)")
        if len(shape) == 2:
            if shape != (N, N):
                raise RuntimeError(f"2D kspring with wrong shape, expected ({N}, {N}), got {shape}")
            if not np.allclose(kspring, kspring.T):
                raise RuntimeError("Spring matrix is not symmetric")

            self._kspring = kspring
        else:  # len(shape) == 0 or len(shape) == 1
            kspring_size = kspring.size
            if kspring_size not in (N, 1):
                raise RuntimeError(f"Wrong kspring size, expected 1 or {N}, got {kspring_size}.")

            self._kspring = np.identity(N) * kspring
        return self._kspring

    @property
    def steer_velocity(self):
        """
        Retrieve current steer_velocity of the collective variable.
        """
        return self._steer_velocity

    @steer_velocity.setter
    def steer_velocity(self, steer_velocity):
        """
        Set the steer_velocity of the collective variable.
        """
        steer_velocity = np.asarray(steer_velocity)
        if steer_velocity.shape == ():
            steer_velocity = steer_velocity.reshape(1)
        if len(steer_velocity.shape) != 1 or steer_velocity.shape[0] != self.cv_dimension:
            raise RuntimeError(
                f"Invalid steer_velocity expected {self.cv_dimension} got {steer_velocity.shape}."
            )
        self._steer_velocity = steer_velocity

    def build(self, snapshot, helpers, *args, **kwargs):
        return _steered(self, snapshot, helpers)


def _steered(method, snapshot, helpers):
    cv = method.cv
    center = method.center
    steer_velocity = method.steer_velocity
    dt = snapshot.dt
    kspring = method.kspring
    natoms = np.size(snapshot.positions, 0)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        centers = center
        work = np.asarray(0.0)
        forces = np.zeros(len(xi))
        return SteeredState(xi, bias, centers, forces, work)

    def update(state, data):
        xi, Jxi = cv(data)
        forces = kspring @ (xi - state.centers).flatten()
        work = state.work + forces @ steer_velocity.flatten() * dt
        centers = state.centers + dt * steer_velocity
        bias = -Jxi.T @ forces.flatten()
        bias = bias.reshape(state.bias.shape)

        return SteeredState(xi, bias, centers, forces, work)

    return snapshot, initialize, generalize(update, helpers)
