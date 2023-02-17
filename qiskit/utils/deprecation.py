# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
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
from typing import Any, Dict, Optional, Type


def deprecate_arguments(
    kwarg_map: Dict[str, str],
    category: Type[Warning] = DeprecationWarning,
    *,
    since: Optional[str] = None,  # pylint: disable=unused-argument
):
    """Decorator to automatically alias deprecated argument names and warn upon use.

    Args:
        kwarg_map: A dictionary of the old argument name to the new name.
        category: Usually either DeprecationWarning or PendingDeprecationWarning.
        since: The version the deprecation started at. Only Optional for backwards
            compatibility - this should always be set.

    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map, category)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_function(
    msg: str,
    stacklevel: int = 2,
    category: Type[Warning] = DeprecationWarning,
    *,
    since: Optional[str] = None,  # pylint: disable=unused-argument
):
    """Emit a warning prior to calling decorated function.

    Args:
        msg: Warning message to emit.
        stacklevel: The warning stacklevel to use, defaults to 2.
        category: Usually either DeprecationWarning or PendingDeprecationWarning.
        since: The version the deprecation started at. Only Optional for backwards
            compatibility - this should always be set.

    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _rename_kwargs(
    func_name: str,
    kwargs: Dict[str, Any],
    kwarg_map: Dict[str, str],
    category: Type[Warning] = DeprecationWarning,
) -> None:
    for old_arg, new_arg in kwarg_map.items():
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError(f"{func_name} received both {new_arg} and {old_arg} (deprecated).")

            if new_arg is None:
                warnings.warn(
                    f"{func_name} keyword argument {old_arg} is deprecated and "
                    "will in future be removed.",
                    category=category,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"{func_name} keyword argument {old_arg} is deprecated and "
                    f"replaced with {new_arg}.",
                    category=category,
                    stacklevel=3,
                )

                kwargs[new_arg] = kwargs.pop(old_arg)
