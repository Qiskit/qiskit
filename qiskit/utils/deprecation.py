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


def deprecate_arg(
    name: str,
    *,
    since: str,
    additional_msg: Optional[str] = None,
    deprecation_description: Optional[str] = None,
    pending: bool = False,
    project_name: str = "Qiskit Terra",
    new_alias: Optional[str] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
):
    """Decorator to indicate an argument has been deprecated in some way.

    This decorator may be used multiple times on the same function, once per deprecated argument.
    It should be placed beneath other decorators like `@staticmethod` and property decorators.

    Args:
        name: The name of the deprecated argument.
        since: The version the deprecation started at. Only Optional for backwards
            compatibility - this should always be set. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update `since` to the new version.
        deprecation_description: What is being deprecated? E.g. "Setting my_func()'s `my_arg`
            argument to `None`." If not set, will default to "{func_name}'s argument `{name}`".
        additional_msg: Put here any additional information, such as what to use instead
            (if new_alias is not set). For example, "Instead, use the argument `new_arg`,
            which is similar but does not impact the circuit's setup."
        pending: Set to `True` if the deprecation is still pending.
        project_name: The name of the project, e.g. "Qiskit Nature".
        new_alias: If the arg has simply been renamed, set this to the new name. The decorator will
            dynamically update the `kwargs` so that when the user sets the old arg, it will be
            passed in as the `new_alias` arg.
        predicate: Only log the runtime warning if the predicate returns True. This is useful to
            deprecate certain values or types for an argument, e.g.
            `lambda my_arg: isinstance(my_arg, dict)`. Regardless of if a predicate is set, the
            runtime warning will only log when the user specifies the argument.

    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        func_name = f"{func.__qualname__}()"  # For methods, `qualname` includes the class name.
        deprecated_entity = deprecation_description or f"{func_name}'s argument `{name}`"
        if pending:
            category = PendingDeprecationWarning
            deprecation_status = "pending deprecation"
            removal_desc = (
                "marked deprecated in a future release, and then removed in a future release"
            )
        else:
            category = DeprecationWarning
            deprecation_status = "deprecated"
            removal_desc = "removed no earlier than 3 months after the release date"

        msg = (
            f"{deprecated_entity} is {deprecation_status} as of {project_name} {since}. "
            f"It will be {removal_desc}."
        )
        if new_alias:
            msg += f" Instead, use the argument `{new_alias}`, which behaves identically."
        if additional_msg:
            msg += f" {additional_msg}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _maybe_warn_and_rename_kwarg(
                    func_name,
                    kwargs,
                    old_arg=name,
                    new_alias=new_alias,
                    warning_msg=msg,
                    category=category,
                    predicate=predicate,
                )
            return func(*args, **kwargs)

        _DeprecationMetadataEntry(msg, since=since, pending=pending).store_on_function(wrapper)
        return wrapper

    return decorator


def deprecate_arguments(
    kwarg_map: Dict[str, Optional[str]],
    category: Type[Warning] = DeprecationWarning,
    *,
    since: Optional[str] = None,
):
    """Deprecated in favor of `deprecate_arg` instead.

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
        func_name = func.__qualname__
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
                for old, new in kwarg_map.items():
                    _maybe_warn_and_rename_kwarg(
                        func_name,
                        kwargs,
                        old_arg=old,
                        new_alias=new,
                        warning_msg=old_kwarg_to_msg[old],
                        category=category,
                        predicate=None,
                    )
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


def _maybe_warn_and_rename_kwarg(
    func_name: str,
    kwargs: Dict[str, Any],
    *,
    old_arg: str,
    new_alias: Optional[str],
    warning_msg: str,
    category: Type[Warning],
    predicate: Optional[Callable[[Any], bool]],
) -> None:
    if old_arg not in kwargs:
        return
    if new_alias and new_alias in kwargs:
        raise TypeError(f"{func_name} received both {new_alias} and {old_arg} (deprecated).")
    if predicate and not predicate(kwargs[old_arg]):
        return
    warnings.warn(warning_msg, category=category, stacklevel=3)
    if new_alias is not None:
        kwargs[new_alias] = kwargs.pop(old_arg)


@dataclass(frozen=True)
class _DeprecationMetadataEntry:
    """Used to store deprecation information on a function.

    This is used by the Qiskit meta repository to render deprecations in documentation. Warning:
    changes may accidentally break the meta repository; pay attention to backwards compatibility.
    """

    msg: str
    since: Optional[str]
    pending: bool

    dunder_name: ClassVar[str] = "__qiskit_deprecations__"

    def store_on_function(self, func: Callable) -> None:
        """Add this metadata to the function's `__qiskit_deprecations__` attribute."""
        if hasattr(func, self.dunder_name):
            getattr(func, self.dunder_name).append(self)
        else:
            setattr(func, self.dunder_name, [self])
