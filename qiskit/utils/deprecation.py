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


def deprecate_arguments(kwarg_map, docstring_version=None):
    """Decorator to automatically alias deprecated argument names and warn upon use."""

    def decorator(func):
        if docstring_version and kwarg_map:
            msg = ["One or more keyword argument are being deprecated:", ""]
            for old_arg, new_arg in kwarg_map.items():
                msg.append("* The argument {} is being replaced with {}".format(old_arg, new_arg))
            msg.append("")  # Finish with an empty line
            _extend_docstring(func, msg, docstring_version)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_function(msg, stacklevel=2, docstring_version=None):
    """Emit a warning prior to calling decorated function.

    Args:
        msg (str): Warning message to emit.
        stacklevel (int): The warning stackevel to use, defaults to 2.
        docstring_version (str): If a version number, extends the docstring with a deprecation warning
           box. If `None`, no docstring box will be added. Default: None

    Returns:
        Callable: The decorated, deprecated callable.
    """

    def decorator(func):
        if docstring_version:
            _extend_docstring(func, msg.expandtabs().splitlines(), docstring_version)

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


def _extend_docstring(func, msg_lines, version):
    docstr = func.__doc__
    if docstr:
        docstr_lines = docstr.expandtabs().splitlines()
    else:
        docstr_lines = ["DEPRECATED"]
    indent = 1000
    first_empty_line = None
    for line_no, line in enumerate(docstr_lines[1:], start=1):
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
        else:
            if first_empty_line is None:
                first_empty_line = line_no
    if first_empty_line is None:
        first_empty_line = len(docstr_lines)
    spaces = ""
    if indent != 1000:
        spaces = " " * indent

    new_doc_str_lines = docstr_lines[:first_empty_line] + [
        "",
        spaces + f".. deprecated:: {version}",
    ]
    for msg_line in msg_lines:
        new_doc_str_lines.append(spaces + "  " + msg_line)
    func.__doc__ = "\n".join(new_doc_str_lines + docstr_lines[first_empty_line:])
