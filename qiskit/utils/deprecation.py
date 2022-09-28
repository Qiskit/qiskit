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
from typing import Type


def deprecate_arguments(
    kwarg_map, category: Type[Warning] = DeprecationWarning, modify_docstring=True, since=None
):
    """Decorator to automatically alias deprecated argument names and warn upon use."""

    def decorator(func):
        if modify_docstring and since is None:
            warnings.warn(
                "Modifying the docstring needs a version. Add parameter `since` with it.",
                stacklevel=2,
            )
        if modify_docstring and since and kwarg_map:
            func.__doc__ = "\n".join(_extend_docstring(func, since, kwarg_map))

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
    modify_docstring: bool = True,
    since: str = None,
):
    """Emit a warning prior to calling decorated function.

    Args:
        msg: Warning message to emit.
        stacklevel: The warning stackevel to use, defaults to 2.
        category: warning category, defaults to DeprecationWarning
        modify_docstring: docstring box will be added. Default: True
        since: If a version number, extends the docstring with a deprecation warning
           box. If `modify_docstring == True`, then mandatory

    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        if modify_docstring and since is None:
            warnings.warn(
                "Modifying the docstring needs a version. Add parameter `since` with it.",
                stacklevel=2,
            )

        if modify_docstring and since:
            func.__doc__ = "\n".join(
                _extend_docstring(func, since, {None: msg.expandtabs().splitlines()})
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _rename_kwargs(func_name, kwargs, kwarg_map, category: Type[Warning] = DeprecationWarning):
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


def _extend_docstring(func, version, kwarg_map):
    """kwarg_map[None] means message in no kwarg, and it will be
    append to the long desscription"""
    docstr = func.__doc__
    if docstr:
        docstr_lines = docstr.expandtabs().splitlines()
    else:
        docstr_lines = ["DEPRECATED"]

    # Mostly based on:
    # https://peps.python.org/pep-0257/#handling-docstring-indentation
    # --v-v-v-v-v-v-v--
    indent = 1000
    pre_args_line = None
    for line_no, line in enumerate(docstr_lines[1:], start=1):
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
        if line.lstrip() == "Args:" and pre_args_line is None:
            pre_args_line = line_no - 1
    if pre_args_line is None:
        pre_args_line = len(docstr_lines)
    spaces = ""
    if indent != 1000:
        spaces = " " * indent
    # --^-^-^-^-^-^-^--

    new_doc_str_lines = docstr_lines[:pre_args_line]
    if None in kwarg_map:
        new_doc_str_lines += ["", spaces + f".. deprecated:: {version}"]
        for msg_line in kwarg_map[None]:
            new_doc_str_lines.append(spaces + "  " + msg_line)

    arg_indent = indent
    args_section = False
    deprecated_arg = False
    for docstr_line in docstr_lines[pre_args_line:]:
        stripped = docstr_line.lstrip()
        current_indent = len(docstr_line) - len(stripped)
        if args_section:
            if deprecated_arg and current_indent == arg_indent:
                new_doc_str_lines.append(docstr_line)
                deprecated_arg = False
            else:
                if not deprecated_arg:
                    for k in kwarg_map.keys():
                        if k is None:
                            continue
                        if stripped.startswith(k):
                            arg_indent = len(docstr_line) - len(stripped)
                            deprecated_arg = True
                            spaces = " " * arg_indent
                            new_doc_str_lines.append(spaces + k + ":")
                            spaces += " " * 4
                            new_doc_str_lines += [
                                spaces + f".. deprecated:: {version}",
                                spaces + f"    The keyword argument ``{k}`` is deprecated.",
                                spaces + f"    Please, use ``{kwarg_map[k]}`` instead.",
                                "",
                            ]
                            break
                    else:
                        new_doc_str_lines.append(docstr_line)
        else:
            new_doc_str_lines.append(docstr_line)
        if docstr_line.lstrip() == "Args:":
            args_section = True
        if args_section and docstr_line.lstrip() == "":
            args_section = False

    return new_doc_str_lines
