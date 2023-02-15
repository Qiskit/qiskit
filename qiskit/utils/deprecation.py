# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
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
from typing import Type, Optional


def deprecate_string_msg(
    version: str,
    old_module: str,
    old_name: str,
    old_type: Optional[str] = "class",
    project_name: Optional[str] = "Qiskit Terra",
    new_module: Optional[str] = None,
    new_name: Optional[str] = None,
    new_type: Optional[str] = None,
    url: Optional[str] = None,
    additional_msg: Optional[str] = None,
):
    """Builds deprecated message.

    Args:
        version: Version to be used
        old_module: Old module to be used
        old_name: Old name to be used
        old_type: Old type to be used, defaults to class
        project_name: project name to use, defaults to Qiskit Terra.
        new_module: New module to be used, if None, old_module is used instead.
        new_name: New name to be used
        new_type: New type to be used, if None, old_type is used instead.
        url: link to further explanations, tutorials
        additional_msg: any additional message

    Returns:
        Message: The build message
    """
    msg = (
        f"The {old_module} {old_name} {old_type} is deprecated as of {project_name} {version} "
        "and will be removed no sooner than 3 months after the release date. "
    )
    if new_name is not None:
        module_str = new_module if new_module is not None else old_module
        type_str = new_type if new_type is not None else old_type
        msg += f"Instead use the {module_str} {new_name} {type_str}. "
    if url is not None:
        msg += f"More details at {url}. "
    if additional_msg is not None:
        msg += f" {additional_msg}"
    return msg


def deprecate_function_msg(
    version: str,
    old_module: str,
    old_name: str,
    old_type: Optional[str] = "class",
    project_name: Optional[str] = "Qiskit Terra",
    new_module: Optional[str] = None,
    new_name: Optional[str] = None,
    new_type: Optional[str] = None,
    url: Optional[str] = None,
    additional_msg: Optional[str] = None,
    stacklevel: int = 2,
    category: Type[Warning] = DeprecationWarning,
):
    """Emit a warning prior to calling decorated function.

    Args:
        version: Version to be used
        old_module: Old module to be used
        old_name: Old name to be used
        old_type: Old type to be used, defaults to class
        project_name: project name to use, defaults to Qiskit Terra.
        new_module: New module to be used, if None, old_module is used instead.
        new_name: New name to be used
        new_type: New type to be used, if None, old_type is used instead.
        url: link to further explanations, tutorials
        additional_msg: any additional message
        stacklevel: The warning stackevel to use, defaults to 2.
        category: warning category, defaults to DeprecationWarning

    Returns:
        Callable: The decorated, deprecated callable.
    """
    msg = deprecate_string_msg(
        version=version,
        old_module=old_module,
        old_name=old_name,
        old_type=old_type,
        project_name=project_name,
        new_module=new_module,
        new_name=new_name,
        new_type=new_type,
        url=url,
        additional_msg=additional_msg,
    )
    return deprecate_function(msg, stacklevel, category)


def deprecate_arguments(kwarg_map, category: Type[Warning] = DeprecationWarning):
    """Decorator to automatically alias deprecated argument names and warn upon use."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map, category)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_function(msg: str, stacklevel: int = 2, category: Type[Warning] = DeprecationWarning):
    """Emit a warning prior to calling decorated function.

    Args:
        msg: Warning message to emit.
        stacklevel: The warning stackevel to use, defaults to 2.
        category: warning category, defaults to DeprecationWarning

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
