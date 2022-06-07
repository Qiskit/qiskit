# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=wrong-import-position, unused-import

"""Alias for Qiskit QPY import."""

import warnings

from qiskit import qpy


deprecated_names = {
    "dump",
    "load",
    # For backward compatibility. Provide, Runtime, Experiment call these private functions.
    "_write_instruction",
    "_read_instruction",
    "_write_parameter_expression",
    "_write_parameter_expression",
    "_read_parameter_expression_v3",
}


def __getattr__(name):
    if name in deprecated_names:
        warnings.warn(
            "The qiskit.circuit.qpy_serialization module is deprecated and has been supersceded "
            f"by the qiskit.qpy module. The {name} function should be accessed from the qiskit.qpy "
            "module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(qpy, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
