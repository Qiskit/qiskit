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
import re
import warnings


def _filter_deprecation_warnings():
    """Apply filters to deprecation warnings.

    Force the `DeprecationWarning` warnings to be displayed for the qiskit
    module, overriding the system configuration as they are ignored by default
    [1] for end-users. Additionally, silence the `ChangedInMarshmallow3Warning`
    messages.

    TODO: on Python 3.7, this might not be needed due to PEP-0565 [2].

    [1] https://docs.python.org/3/library/warnings.html#default-warning-filters
    [2] https://www.python.org/dev/peps/pep-0565/
    """
    deprecation_filter = (
        "default",
        None,
        DeprecationWarning,
        re.compile(r"^qiskit\.*", re.UNICODE),
        0,
    )

    # Instead of using warnings.simple_filter() directly, the internal
    # _add_filter() function is used for being able to match against the
    # module.
    try:
        warnings._add_filter(*deprecation_filter, append=False)
    except AttributeError:
        # ._add_filter is internal and not available in some Python versions.
        pass


_filter_deprecation_warnings()


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
            # warn only once
            if not wrapper._warned:
                warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
                wrapper._warned = True
            return func(*args, **kwargs)

        wrapper._warned = False
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
