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

"""Approximate Quantum Compiler - to be deprecated"""

import warnings

from .approximate import ApproximateCircuit, ApproximatingObjective
from .aqc import AQC
from .cnot_structures import make_cnot_network
from .cnot_unit_circuit import CNOTUnitCircuit
from .cnot_unit_objective import CNOTUnitObjective, DefaultCNOTUnitObjective
from .fast_gradient.fast_gradient import FastCNOTUnitObjective
from .aqc_plugin import AQCSynthesisPlugin

warnings.warn(
    "The qiskit.transpiler.synthesis.aqc module is pending deprecation since Qiskit 0.46.0. "
    "It will be deprecated in a following release, no sooner than 3 months after the 0.46.0 release.",
    stacklevel=2,
    category=PendingDeprecationWarning,
)
