# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from copy import deepcopy
from functools import partial
from typing import Callable, NamedTuple

from jax import jit, numpy as np
from jaxlib.xla_extension import GpuDevice

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

import importlib


class Sampler:
    def __init__(self, method_bundle, context_state):
        snapshot, initialize, update = method_bundle
        self.context_state = context_state
        self.snapshot = snapshot
        self.state = initialize()
        self.update = update


def take_snapshot(state, box, dt):
    positions = state.position
    forces = getattr(state, "force", state.position)
    ids = np.arange(len(positions))
    velocities = getattr(state, "velocity", state.position)
    masses = state.mass.reshape(-1, 1)
    vel_mass = (velocities, masses)
    origin = (0.0, 0.0, 0.0)
    return Snapshot(positions, vel_mass, forces, ids, None, Box(box, origin), dt)


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
    if isinstance(context.box.device(), GpuDevice):
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


def build_runner(context, sampler, callback):
    box = context.box
    dt = context.dt

    def run(timesteps):
        for i in range(timesteps):
            context_state = sampler.context_state
            sampler.snapshot = take_snapshot(context_state, box, dt)
            sampler.state = sampler.update(sampler.snapshot, sampler.state)
            sampler.context_state = context.step_fn(context_state, sampler.state.bias)
            if callback:
                callback(sampler.snapshot, sampler.state, i)

    return run


class View(NamedTuple):
    synchronize: Callable


def bind(
    wrapped_context: ContextWrapper, sampling_method: SamplingMethod, callback: Callable, **kwargs
):
    context = wrapped_context.context
    context_state = context.init_fn(**kwargs)
    snapshot = take_snapshot(context_state, context.box, context.dt)
    helpers = build_helpers(wrapped_context.view, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle, context_state)
    wrapped_context.view = View((lambda: None))
    wrapped_context.run = build_runner(context, sampler, callback)
    return sampler