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


"""Replace each sequence of Clifford gates by a single Clifford gate."""

from qiskit.circuit.library.standard_gates import (
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    CXGate,
    CYGate,
    CZGate,
    SwapGate,
)
from qiskit.quantum_info.operators import Clifford
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.optimization.collapse_chains import CollapseChains


clifford_gate_name_to_gate_class = {
    "x": XGate,
    "y": YGate,
    "z": ZGate,
    "h": HGate,
    "s": SGate,
    "sdg": SdgGate,
    "cx": CXGate,
    "cy": CYGate,
    "cz": CZGate,
    "swap": SwapGate,
}


class CollectCliffords(CollapseChains):
    """Collects blocks of Clifford gates and replaces them by a :class:`~qiskit.quantum_info.Clifford`
    object.
    """

    def filter_function(self, node):
        """Specifies which nodes to collect into blocks."""
        return (
            node.op.name in clifford_gate_name_to_gate_class.keys()
            and getattr(node.op, "condition", None) is None
        )

    def collapse_function(self, qc: QuantumCircuit):
        """Specifies how to construct an Operation from a collected block."""
        return Clifford(qc)
