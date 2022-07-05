# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from typing import Callable, NamedTuple

from jax import default_backend as default_device, jit, numpy as np

from pysages.backends.core import ContextWrapper
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
    restore as _restore,
)
from pysages.methods import SamplingMethod
from pysages.utils import ToCPU, copy

import importlib


class Sampler:
    def __init__(self, method_bundle):
        snapshot, initialize, update = method_bundle
        self.snapshot = snapshot
        self.state = initialize()
        self.update = update


def take_snapshot(simulation):
    atoms = simulation.atoms
    #
    positions = np.asarray(atoms.get_positions())
    forces = np.asarray(atoms.get_forces(md=True))
    ids = np.arange(atoms.get_global_number_of_atoms())
    #
    velocities = np.asarray(atoms.get_velocities())
    masses = np.asarray(atoms.get_masses()).reshape(-1, 1)
    vel_mass = (velocities, masses)
    #
    a = atoms.cell[0]
    b = atoms.cell[1]
    c = atoms.cell[2]
    H = ((a[0], b[0], c[0]), (a[1], b[1], c[1]), (a[2], b[2], c[2]))
    origin = (0.0, 0.0, 0.0)
    dt = simulation.dt
    # ASE doesn't use images explicitely
    return Snapshot(positions, vel_mass, forces, ids, None, Box(H, origin), dt)


def build_snapshot_methods(context, sampling_method):
    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        V, M = snapshot.vel_mass
        return (V * M).flatten()

    return SnapshotMethods(jit(positions), jit(indices), jit(momenta), jit(masses))


def build_helpers(context, sampling_method):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if default_device() == "gpu":
        cupy = importlib.import_module("cupy")
        view = cupy.asarray
    else:
        utils = importlib.import_module(".utils", package="pysages.backends")
        view = utils.view

    def restore_vm(view, snapshot, prev_snapshot):
        # TODO: Check if we can omit modifying the masses
        # (in general the masses are unlikely to change)
        velocities = view(snapshot.vel_mass[0])
        masses = view(snapshot.vel_mass[1])
        velocities[:] = view(prev_snapshot.vel_mass[0])
        masses[:] = view(prev_snapshot.vel_mass[1])

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view, restore_vm=restore_vm)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), restore)

    return helpers


def wrap_step_fn(simulation, sampler, callback):
    """
    Wraps the original step function of the `ase.md.MolecularDynamics`
    instance `simulation`, and injects calls to the sampling method's `update`
    and the user provided callback.
    """
    number_of_steps = simulation.get_number_of_steps
    simulation._step = simulation.step

    def wrapped_step():
        sampler.snapshot = take_snapshot(simulation)
        sampler.state = sampler.update(sampler.snapshot, sampler.state)
        forces = copy(sampler.snapshot.forces + sampler.state.bias, ToCPU())
        simulation._step(forces=forces)
        if callback:
            callback(sampler.snapshot, sampler.state, number_of_steps())

    simulation.step = wrapped_step


class View(NamedTuple):
    synchronize: Callable


def bind(
    wrapped_context: ContextWrapper, sampling_method: SamplingMethod, callback: Callable, **kwargs
):
    """
    Entry point for the backend code, it gets called when the simulation
    context is wrapped within `pysages.run`.
    """
    context = wrapped_context.context
    snapshot = take_snapshot(context)
    helpers = build_helpers(wrapped_context.view, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle)
    wrapped_context.view = View((lambda: None))
    wrap_step_fn(context, sampler, callback)
    wrapped_context.run = context.run
    return sampler
