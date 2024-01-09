# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================================================
Approximate Quantum Compiler (:mod:`qiskit.transpiler.synthesis.aqc`)
=====================================================================
"""

import warnings

from qiskit.synthesis.unitary.aqc.approximate import ApproximateCircuit, ApproximatingObjective
from qiskit.synthesis.unitary.aqc import AQC
from qiskit.synthesis.unitary.aqc.cnot_structures import make_cnot_network
from qiskit.synthesis.unitary.aqc.cnot_unit_circuit import CNOTUnitCircuit
from qiskit.synthesis.unitary.aqc.cnot_unit_objective import (
    CNOTUnitObjective,
    DefaultCNOTUnitObjective,
)
from qiskit.synthesis.unitary.aqc.fast_gradient.fast_gradient import FastCNOTUnitObjective
from qiskit.transpiler.passes.synthesis.aqc_plugin import AQCSynthesisPlugin

warnings.warn(
    "The qiskit.transpiler.synthesis.aqc module is pending deprecation since Qiskit 0.46.0. "
    "It will be deprecated in a following release, no sooner than 3 months after the 0.46.0 release.",
    stacklevel=2,
    category=PendingDeprecationWarning,
)

_DEPRECATED_NAMES = {
    "AQC": "qiskit.synthesis.unitary.aqc",
    "ApproximateCircuit": "qiskit.synthesis.unitary.aqc",
    "ApproximatingObjective": "qiskit.synthesis.unitary.aqc",
    "CNOTUnitCircuit": "qiskit.synthesis.unitary.aqc",
    "CNOTUnitObjective": "qiskit.synthesis.unitary.aqc",
    "DefaultCNOTUnitObjective": "qiskit.synthesis.unitary.aqc",
    "FastCNOTUnitObjective": "qiskit.synthesis.unitary.aqc",
    "AQCSynthesisPlugin": "qiskit.transpiler.passes.synthesis",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        import importlib

        module_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"Accessing '{name}' from '{__name__}' is deprecated since Qiskit 0.46"
            f" and will be removed in 1.0.  Import from '{module_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_name), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
