# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Deprecation utilities"""

import functools
import warnings


def deprecate_arguments(kwarg_map):
    """Decorator to automatically alias deprecated argument names and warn upon use."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_function(msg, stacklevel=2):
    """Emit a warning prior to calling decorated function.

    Args:
        msg (str): Warning message to emit.
        stacklevel (int): The warning stackevel to use, defaults to 2.

    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _rename_kwargs(func_name, kwargs, kwarg_map):
    for old_arg, new_arg in kwarg_map.items():
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError(f"{func_name} received both {new_arg} and {old_arg} (deprecated).")

            warnings.warn(
                "{} keyword argument {} is deprecated and "
                "replaced with {}.".format(func_name, old_arg, new_arg),
                DeprecationWarning,
                stacklevel=3,
            )

            kwargs[new_arg] = kwargs.pop(old_arg)
