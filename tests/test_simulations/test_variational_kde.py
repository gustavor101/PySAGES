# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Unit and integration tests for the four online-deposited variational kernel
enhanced sampling methods (`DBKMetaD`, `FOKMetaD`, `DBKABF`, `DBCZAR`).
"""

import tempfile
from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

import pysages
from pysages.colvars import Component
from pysages.methods import (
    CVRestraints,
    DBCZAR,
    DBKABF,
    DBKMetaD,
    DepositedBatchCZAR,
    DepositedBatchKernelABF,
    DepositedBatchKernelMetaD,
    FirstOrderKernelMetaD,
    FOKMetaD,
)


def _generate_2d_harmonic_sim(T=300.0, dt_ps=0.002, **kwargs):
    system = mm.System()
    system.addParticle(12.0 * unit.dalton)
    topology = app.Topology()
    chain = topology.addChain("A")
    res = topology.addResidue("P", chain)
    topology.addAtom("C", app.Element.getBySymbol("C"), res)

    force = mm.CustomExternalForce("50.0 * ((x - 0.1)^2 + (y + 0.1)^2 + z^2)")
    force.addParticle(0, [])
    system.addForce(force)

    integrator = mm.LangevinMiddleIntegrator(
        T * unit.kelvin, 5.0 / unit.picosecond, dt_ps * unit.picoseconds
    )
    platform = mm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions([mm.Vec3(0.0, 0.0, 0.0)] * unit.nanometer)
    sim.context.setVelocitiesToTemperature(T * unit.kelvin)
    return sim


def test_aliases():
    assert DepositedBatchKernelMetaD is DBKMetaD
    assert FirstOrderKernelMetaD is FOKMetaD
    assert DepositedBatchKernelABF is DBKABF
    assert DepositedBatchCZAR is DBCZAR


@pytest.mark.parametrize(
    "method_cls",
    [DBKMetaD, FOKMetaD, DBKABF, DBCZAR],
)
@pytest.mark.parametrize("with_grid", [True, False])
def test_online_deposited_kernel_methods(method_cls, with_grid):
    cvs = [Component([0], axis=0), Component([0], axis=1)]
    grid = (
        pysages.Grid(lower=(-1.0, -1.0), upper=(1.0, 1.0), shape=(16, 16), periodic=False)
        if with_grid
        else None
    )
    restraints = CVRestraints((-1.0, -1.0), (1.0, 1.0), (200.0, 200.0), (200.0, 200.0))

    method = method_cls(
        cvs=cvs,
        sigma=(0.15, 0.15),
        stride=20,
        max_kernels=15,
        deltaT=1200.0,
        T=300.0,
        sigma_min=(0.10, 0.10),
        grid=grid,
        restraints=restraints,
    )

    result = pysages.run(method, _generate_2d_harmonic_sim, 100, context_args={})
    ana = pysages.analyze(result)

    assert ana["n_kernels"] == 5
    assert ana["centers"].shape == (5, 2)
    assert np.all(np.isfinite(ana["free_energy"]))
    assert np.isclose(np.min(ana["free_energy"]), 0.0)

    probe_pts = np.array([[0.0, 0.0], [0.1, -0.1], [-0.2, 0.2]])
    probe_fe = np.asarray(ana["fes_fn"](probe_pts))
    assert probe_fe.shape == (3,)
    assert np.all(np.isfinite(probe_fe))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "state.pkl"
        pysages.save(result, str(ckpt))
        loaded = pysages.load(str(ckpt))
        ana_loaded = pysages.analyze(loaded)
        assert ana_loaded["n_kernels"] == 5
        assert np.allclose(ana_loaded["free_energy"], ana["free_energy"])
