# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utils for testing the standard gates."""

from inspect import signature, Parameter


def _get_free_params(fun, ignore=None):
    """Get the names of the free parameters of the function ``f``.

    Args:
        fun (callable): The function to inspect.
        ignore (list[str]): A list of argument names (as str) to ignore.

    Returns:
        list[str]: The name of the free parameters not listed in ``ignore``.
    """
    ignore = ignore or []
    free_params = []
    for name, param in signature(fun).parameters.items():
        if param.default == Parameter.empty and param.kind != Parameter.VAR_POSITIONAL:
            if name not in ignore:
                free_params.append(name)
    return free_params
