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
from typing import Callable, ClassVar, Dict, Optional, Type, cast, Any


def deprecate_arguments(kwarg_map: Dict[str, str], category: Type[Warning] = DeprecationWarning):
    """Decorator to automatically alias deprecated argument names and warn upon use."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func, kwargs, kwarg_map, category)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_function(msg: str, stacklevel: int = 2, category: Type[Warning] = DeprecationWarning):
    """Emit a warning prior to calling decorated function.

    Args:
        msg: Warning message to emit.
        stacklevel: The warning stacklevel to use, defaults to 2.
        category: warning category, defaults to DeprecationWarning

    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        _DeprecationMetadata.set_func_deprecation(func, msg=msg, since="TODO")
        return wrapper

    return decorator


def _rename_kwargs(
    func: Callable,
    kwargs: Dict[str, Any],
    kwarg_map: Dict[str, str],
    category: Type[Warning] = DeprecationWarning,
) -> None:
    func_name = func.__name__
    for old_arg, new_arg in kwarg_map.items():
        if new_arg is None:
            msg = (
                f"{func_name} keyword argument {old_arg} is deprecated and "
                "will in the future be removed."
            )
        else:
            msg = (
                f"{func_name} keyword argument {old_arg} is deprecated and "
                f"replaced with {new_arg}."
            )
        _DeprecationMetadata.set_args_deprecation(func, arg=old_arg, msg=msg, since="TODO")
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError(f"{func_name} received both {new_arg} and {old_arg} (deprecated).")
            warnings.warn(msg, category=category, stacklevel=3)
            if new_arg is not None:
                kwargs[new_arg] = kwargs.pop(old_arg)


@dataclass(frozen=True)
class _DeprecationMetadataEntry:
    msg: str
    since: str


@dataclass
class _DeprecationMetadata:
    """Used to store deprecation information on a function.

    This is used by the Qiskit Sphinx Theme to render deprecations in documentation. Warning:
    coordinate changes with the Sphinx Theme's extension.
    """

    func_deprecation: Optional[_DeprecationMetadataEntry]
    args_deprecations: Dict[str, _DeprecationMetadataEntry]

    dunder_name: ClassVar[str] = "__qiskit_deprecation__"

    @classmethod
    def set_func_deprecation(cls, func: Callable, *, msg: str, since: str) -> None:
        entry = _DeprecationMetadataEntry(msg, since)
        if hasattr(func, cls.dunder_name):
            metadata = cast(_DeprecationMetadata, getattr(func, cls.dunder_name))
            metadata.func_deprecation = entry
        else:
            metadata = cls(func_deprecation=entry, args_deprecations={})
            setattr(func, cls.dunder_name, metadata)

    @classmethod
    def set_args_deprecation(cls, func: Callable, *, arg: str, msg: str, since: str) -> None:
        entry = _DeprecationMetadataEntry(msg, since)
        if hasattr(func, cls.dunder_name):
            metadata = cast(_DeprecationMetadata, getattr(func, cls.dunder_name))
            metadata.args_deprecations[arg] = entry
        else:
            metadata = cls(func_deprecation=None, args_deprecations={arg: entry})
            setattr(func, cls.dunder_name, metadata)
