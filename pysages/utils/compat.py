# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from importlib import import_module

from jax.scipy import linalg

from pysages._compat import (
    _jax_version_tuple,
    _plum_version_tuple,
    _python_version_tuple,
)

# Compatibility utils


def try_import(new_name, old_name):
    try:
        return import_module(new_name)
    except ModuleNotFoundError:
        return import_module(old_name)


if _python_version_tuple >= (3, 8):
    prod = import_module("math").prod
else:

    def prod(iterable, start=1):
        """
        Calculate the product of all the elements in the input iterable.
        When the iterable is empty, return the start value (1 by default).
        """
        result = start
        for x in iterable:
            result *= x
        return result


# Compatibility for jax >=0.4.31

# https://github.com/google/jax/blob/main/CHANGELOG.md#jax-0431-july-29-2024
if _jax_version_tuple < (0, 4, 31):
    _jax_core = import_module("jax.core")

    def canonicalize_shape(shape):
        return _jax_core.as_named_shape(shape)

else:
    _jax_core = import_module("jax._src.core")

    def canonicalize_shape(shape):
        return _jax_core.canonicalize_shape(shape)


# Compatibility for jax >=0.4.22

# https://github.com/google/jax/blob/main/CHANGELOG.md#jax-0422-dec-13-2023
if _jax_version_tuple < (0, 4, 22):

    def unsafe_buffer_pointer(array):
        return array.device_buffer.unsafe_buffer_pointer()

else:

    def unsafe_buffer_pointer(array):
        return array.unsafe_buffer_pointer()


# Compatibility for jax >=0.4.21

# https://github.com/google/jax/blob/main/CHANGELOG.md#jax-0421-dec-4-2023
if _jax_version_tuple < (0, 4, 21):

    def device_platform(array):
        return array.device().platform

else:

    def device_platform(array):
        return next(iter(array.devices())).platform


# Compatibility for jax >=0.4.1

# https://github.com/google/jax/releases/tag/jax-v0.4.1
if _jax_version_tuple < (0, 4, 1):

    def check_device_array(array):
        pass

else:

    def check_device_array(array):
        if not (array.is_fully_addressable and len(array.sharding.device_set) == 1):
            err = "Support for SharedDeviceArray or GlobalDeviceArray has not been implemented"
            raise ValueError(err)


# Compatibility for jax >=0.3.15

# https://github.com/google/jax/compare/jaxlib-v0.3.14...jax-v0.3.15
# https://github.com/google/jax/pull/11546
if _jax_version_tuple < (0, 3, 15):

    def solve_pos_def(a, b):
        return linalg.solve(a, b, sym_pos="sym")

else:

    def solve_pos_def(a, b):
        return linalg.solve(a, b, assume_a="pos")


# Compatibility for plum >=2

# https://github.com/beartype/plum/pull/73
if _plum_version_tuple < (2, 0, 0):

    def dispatch_table(dispatch):
        return dispatch._functions

    def has_method(fn, T, index):
        types_at_index = set()
        for sig in fn.methods.keys():
            types_at_index.update(sig.types[index].get_types())
        return T in types_at_index

    is_generic_subclass = issubclass

else:
    _bt = import_module("beartype.door")
    _typing = import_module("plum" if _plum_version_tuple < (2, 2, 1) else "typing")
    _util = _typing.type if _plum_version_tuple < (2, 2, 1) else _typing

    if _plum_version_tuple < (2, 3, 0):

        def _signature_types(sig):
            return sig.types

    else:

        def _signature_types(sig):
            return sig.signature.types

    def dispatch_table(dispatch):
        return dispatch.functions

    def has_method(fn, T, index):
        types_at_index = set()
        for sig in fn.methods:
            typ = _signature_types(sig)[index]
            if _util.get_origin(typ) is _typing.Union:
                types_at_index.update(_util.get_args(typ))
            else:
                types_at_index.add(typ)
        return T in types_at_index

    def is_generic_subclass(A, B):
        return _bt.TypeHint(A) <= _bt.TypeHint(B)
