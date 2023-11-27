# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
===============================
Utilities (:mod:`qiskit.utils`)
===============================

.. currentmodule:: qiskit.utils

.. autofunction:: add_deprecation_to_docstring
.. autofunction:: deprecate_arg
.. autofunction:: deprecate_arguments
.. autofunction:: deprecate_func
.. autofunction:: deprecate_function
.. autofunction:: local_hardware_info
.. autofunction:: is_main_process
.. autofunction:: apply_prefix
.. autofunction:: detach_prefix
.. autofunction:: wrap_method

Algorithm Utilities
===================

.. autofunction:: summarize_circuits
.. autofunction:: get_entangler_map
.. autofunction:: validate_entangler_map
.. autofunction:: name_args


Optional Dependency Checkers (:mod:`qiskit.utils.optionals`)
============================================================

.. automodule:: qiskit.utils.optionals
"""

from .deprecation import (
    add_deprecation_to_docstring,
    deprecate_arg,
    deprecate_arguments,
    deprecate_func,
    deprecate_function,
)
from .multiprocessing import local_hardware_info
from .multiprocessing import is_main_process
from .units import apply_prefix, detach_prefix
from .classtools import wrap_method
from .lazy_tester import LazyDependencyManager, LazyImportTester, LazySubprocessTester

from . import optionals

from .circuit_utils import summarize_circuits
from .entangler_map import get_entangler_map, validate_entangler_map
from .name_unnamed_args import name_args
from .experimental import ExperimentalQiskitAPI

__all__ = [
    "LazyDependencyManager",
    "LazyImportTester",
    "LazySubprocessTester",
    "summarize_circuits",
    "get_entangler_map",
    "validate_entangler_map",
    "name_args",
    "add_deprecation_to_docstring",
    "deprecate_arg",
    "deprecate_arguments",
    "deprecate_func",
    "deprecate_function",
    "local_hardware_info",
    "is_main_process",
    "apply_prefix",
    "ExperimentalQiskitAPI",
]
