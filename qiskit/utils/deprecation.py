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
from typing import Any, Callable, Dict, Optional, Type


def deprecate_arguments(
    kwarg_map: Dict[str, Optional[str]],
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

    del since  # Will be used in a followup to add deprecations to our docs site.

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

    del since  # Will be used in a followup to add deprecations to our docs site.

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


# We insert deprecations in-between the description and Napoleon's meta sections. The below is from
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#docstring-sections. We use
# lowercase because Napoleon is case-insensitive.
_NAPOLEON_META_LINES = frozenset(
    {
        "args:",
        "arguments:",
        "attention:",
        "attributes:",
        "caution:",
        "danger:",
        "error:",
        "example:",
        "examples:",
        "hint:",
        "important:",
        "keyword args:",
        "keyword arguments:",
        "note:",
        "notes:",
        "other parameters:",
        "parameters:",
        "return:",
        "returns:",
        "raises:",
        "references:",
        "see also:",
        "tip:",
        "todo:",
        "warning:",
        "warnings:",
        "warn:",
        "warns:",
        "yield:",
        "yields:",
    }
)


def add_deprecation_to_docstring(
    func: Callable, msg: str, *, since: Optional[str], pending: bool
) -> None:
    """Dynamically insert the deprecation message into ``func``'s docstring.

    Args:
        func: The function to modify.
        msg: The full deprecation message.
        since: The version the deprecation started at.
        pending: Is the deprecation still pending?
    """
    if "\n" in msg:
        raise ValueError(
            "Deprecation messages cannot contain new lines (`\\n`), but the deprecation for "
            f'{func.__qualname__} had them. Usually this happens when using `"""` multiline '
            f"strings; instead, use string concatenation.\n\n"
            "This is a simplification to facilitate deprecation messages being added to the "
            "documentation. If you have a compelling reason to need "
            "new lines, feel free to improve this function or open a request at "
            "https://github.com/Qiskit/qiskit-terra/issues."
        )

    if since is None:
        version_str = "unknown"
    else:
        version_str = f"{since}_pending" if pending else since

    indent = ""
    meta_index = None
    if func.__doc__:
        original_lines = func.__doc__.splitlines()
        content_encountered = False
        for i, line in enumerate(original_lines):
            stripped = line.strip()

            # Determine the indent based on the first line with content. But, we don't consider the
            # first line, which corresponds to the format """Docstring.""", as it does not properly
            # capture the indentation of lines beneath it.
            if not content_encountered and i != 0 and stripped:
                num_leading_spaces = len(line) - len(line.lstrip())
                indent = " " * num_leading_spaces
                content_encountered = True

            if stripped.lower() in _NAPOLEON_META_LINES:
                meta_index = i
                if content_encountered is not True:
                    raise ValueError(
                        "add_deprecation_to_docstring cannot currently handle when a Napoleon "
                        "metadata line like 'Args' is the very first line of docstring, "
                        f'e.g. `"""Args:`. So, it cannot process {func.__qualname__}. Instead, '
                        f'move the metadata line to the second line, e.g.:\n\n"""\nArgs:'
                    )
                # We can stop checking since we only care about the first meta line, and
                # we've validated content_encountered is True to determine the indent.
                break
    else:
        original_lines = []

    # We defensively include new lines in the beginning and end. This is sometimes necessary,
    # depending on the original docstring. It is not a big deal to have extra, other than `help()`
    # being a little longer.
    new_lines = [
        indent,
        f"{indent}.. deprecated:: {version_str}",
        f"{indent}  {msg}",
        indent,
    ]

    if meta_index:
        original_lines[meta_index - 1 : meta_index - 1] = new_lines
    else:
        original_lines.extend(new_lines)
    func.__doc__ = "\n".join(original_lines)
