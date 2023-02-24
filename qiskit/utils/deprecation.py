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
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Optional, Type


def deprecate_arguments(
    kwarg_map: Dict[str, str],
    category: Type[Warning] = DeprecationWarning,
    *,
    since: Optional[str] = None,
):
    """Decorator to automatically alias deprecated argument names and warn upon use.

    Args:
        kwarg_map: A dictionary of the old argument name to the new name.
        category: Usually either DeprecationWarning or PendingDeprecationWarning.
        since: The version the deprecation started at. Only Optional for backwards
            compatibility - this should always be set. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update `since` to the new version.

    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        func_name = func.__name__
        old_kwarg_to_msg = {}
        for old_arg, new_arg in kwarg_map.items():
            msg_suffix = (
                "will in the future be removed." if new_arg is None else f"replaced with {new_arg}."
            )
            old_kwarg_to_msg[
                old_arg
            ] = f"{func_name} keyword argument {old_arg} is deprecated and {msg_suffix}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func_name, kwargs, old_kwarg_to_msg, kwarg_map, category)
            return func(*args, **kwargs)

        for msg in old_kwarg_to_msg.values():
            _DeprecationMetadataEntry(
                msg, since=since, pending=issubclass(category, PendingDeprecationWarning)
            ).store_on_function(wrapper)
        return wrapper

    return decorator


def deprecate_function(
    msg: str,
    stacklevel: int = 2,
    category: Type[Warning] = DeprecationWarning,
    *,
    since: Optional[str] = None,
):
    """Emit a warning prior to calling decorated function.

    Args:
        msg: Warning message to emit.
        stacklevel: The warning stacklevel to use, defaults to 2.
        category: Usually either DeprecationWarning or PendingDeprecationWarning.
        since: The version the deprecation started at. Only Optional for backwards
            compatibility - this should always be set. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update `since` to the new version.

    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        _DeprecationMetadataEntry(
            msg=msg, since=since, pending=issubclass(category, PendingDeprecationWarning)
        ).store_on_function(wrapper)
        return wrapper

    return decorator


def _rename_kwargs(
    func_name: str,
    kwargs: Dict[str, Any],
    old_kwarg_to_msg: Dict[str, str],
    kwarg_map: Dict[str, str],
    category: Type[Warning] = DeprecationWarning,
) -> None:
    for old_arg, new_arg in kwarg_map.items():
        if old_arg not in kwargs:
            continue
        if new_arg in kwargs:
            raise TypeError(f"{func_name} received both {new_arg} and {old_arg} (deprecated).")
        warnings.warn(old_kwarg_to_msg[old_arg], category=category, stacklevel=3)
        if new_arg is not None:
            kwargs[new_arg] = kwargs.pop(old_arg)


@dataclass(frozen=True)
class _DeprecationMetadataEntry:
    """Used to store deprecation information on a function.

    This is used by the Qiskit meta repository to render deprecations in documentation. Warning:
    changes may accidentally break the meta repository; pay attention to backwards compatibility.
    """

    msg: str
    since: str
    pending: bool

    dunder_name: ClassVar[str] = "__qiskit_deprecations__"

    def store_on_function(self, func: Callable) -> None:
        """Add this metadata to the function's `__qiskit_deprecations__` attribute."""
        if hasattr(func, self.dunder_name):
            getattr(func, self.dunder_name).append(self)
        else:
            setattr(func, self.dunder_name, [self])
