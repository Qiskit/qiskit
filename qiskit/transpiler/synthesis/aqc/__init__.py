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
============================
Approximate Quantum Compiler
============================
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
    "The qiskit.transpiler.synthesis.aqc module is deprecated since Qiskit 0.46.0 "
    "and will be removed in Qiskit 1.0.",
    stacklevel=2,
    category=DeprecationWarning,
)
