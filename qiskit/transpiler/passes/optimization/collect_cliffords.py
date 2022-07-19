# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Replace each sequence of Clifford gates by a single Clifford gate."""

from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate, HGate, SGate, SdgGate, CXGate, CYGate, CZGate, SwapGate
from qiskit.quantum_info.operators import Clifford
from qiskit.circuit import QuantumCircuit
from .collapse_chains import CollapseChains


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
    """Collect blocks of Clifford gates and replaces them by a single Clifford."""

    def filter_function(self, node):
        """Specifies which nodes to collect into blocks."""
        return node.op.name in clifford_gate_name_to_gate_class.keys() and node.op.condition is None

    def collapse_function(self, blocks, dag):
        """Specifies what to do with collected blocks."""
        # Replace every discovered block by a Clifford
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for block in blocks:
            # Find the set of all qubits used in this block
            cur_qubits = set()
            for node in block:
                cur_qubits.update(node.qargs)

            # For reproducibility, order these qubits compatibly with the global order
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))

            # Construct a linear circuit
            qc = QuantumCircuit(len(cur_qubits))
            for node in block:
                positions = [wire_pos_map[qarg] for qarg in node.qargs]
                qc.append(clifford_gate_name_to_gate_class[node.op.name](), positions)

            # Create a Clifford from this quantum circuit
            # Before the PR on Cliffords as HLOs in merged, we unfortunately need to convert
            # Clifford to instruction
            op = Clifford(qc).to_instruction()

            # Replace the block by the constructed circuit
            dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)

        return dag
