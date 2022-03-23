# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Replace each block of consecutive gates by a single Unitary node."""

from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.synthesis import unitary_synthesis


class ConsolidateBlocks(TransformationPass):
    """Replace each block of consecutive gates by a single Unitary node.

    Pass to consolidate sequences of uninterrupted gates acting on
    the same qubits into a Unitary node, to be resynthesized later,
    to a potentially more optimal subcircuit.

    Notes:
        This pass assumes that the 'blocks_list' property that it reads is
        given such that blocks are in topological order. The blocks are
        collected by a previous pass, such as `Collect2qBlocks`.
    """

    def __init__(self, kak_basis_gate=None, force_consolidate=False, basis_gates=None, target=None):
        """ConsolidateBlocks initializer.

        Args:
            kak_basis_gate (Gate): Basis gate for KAK decomposition.
            force_consolidate (bool): Force block consolidation
            basis_gates (List(str)): Basis gates from which to choose a KAK gate.
            target (Target): The target object for the compilation target backend
        """
        super().__init__()
        self.basis_gates = None
        self.target = target
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)
        self.force_consolidate = force_consolidate

        if kak_basis_gate is not None:
            self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
        elif basis_gates is not None:
            self.decomposer = unitary_synthesis._basis_gates_to_decomposer_2q(basis_gates)
        else:
            self.decomposer = TwoQubitBasisDecomposer(CXGate())

    def run(self, dag):
        """Run the ConsolidateBlocks pass on `dag`.

        Iterate over each block and replace it with an equivalent Unitary
        on the same wires.
        """
        if self.decomposer is None:
            return dag

        # compute ordered indices for the global circuit wires
        global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        blocks = self.property_set["block_list"]
        basis_gate_name = self.decomposer.gate.name
        all_block_gates = set()
        for block in blocks:
            if len(block) == 1 and self._check_not_in_basis(
                block[0].name, block[0].qargs, global_index_map
            ):
                all_block_gates.add(block[0])
                dag.substitute_node(block[0], UnitaryGate(block[0].op.to_matrix()))
            else:
                basis_count = 0
                outside_basis = False
                block_qargs = set()
                block_cargs = set()
                for nd in block:
                    block_qargs |= set(nd.qargs)
                    if isinstance(nd, DAGOpNode) and nd.op.condition:
                        block_cargs |= set(nd.op.condition[0])
                    all_block_gates.add(nd)
                q = QuantumRegister(len(block_qargs))
                qc = QuantumCircuit(q)
                if block_cargs:
                    c = ClassicalRegister(len(block_cargs))
                    qc.add_register(c)
                block_index_map = self._block_qargs_to_indices(block_qargs, global_index_map)
                for nd in block:
                    if nd.op.name == basis_gate_name:
                        basis_count += 1
                    if self._check_not_in_basis(nd.op.name, nd.qargs, global_index_map):
                        outside_basis = True
                    qc.append(nd.op, [q[block_index_map[i]] for i in nd.qargs])
                unitary = UnitaryGate(Operator(qc))

                max_2q_depth = 20  # If depth > 20, there will be 1q gates to consolidate.
                if (  # pylint: disable=too-many-boolean-expressions
                    self.force_consolidate
                    or unitary.num_qubits > 2
                    or self.decomposer.num_basis_gates(unitary) < basis_count
                    or len(block) > max_2q_depth
                    or ((self.basis_gates is not None) and outside_basis)
                    or ((self.target is not None) and outside_basis)
                ):
                    dag.replace_block_with_op(block, unitary, block_index_map, cycle_check=False)
        # If 1q runs are collected before consolidate those too
        runs = self.property_set["run_list"] or []
        for run in runs:
            if run[0] in all_block_gates:
                continue
            if len(run) == 1 and not self._check_not_in_basis(
                run[0].name, run[0].qargs, global_index_map
            ):
                dag.substitute_node(run[0], UnitaryGate(run[0].op.to_matrix()))
            else:
                qubit = run[0].qargs[0]
                operator = run[0].op.to_matrix()
                already_in_block = False
                for gate in run[1:]:
                    if gate in all_block_gates:
                        already_in_block = True
                    operator = gate.op.to_matrix().dot(operator)
                if already_in_block:
                    continue
                unitary = UnitaryGate(operator)
                dag.replace_block_with_op(run, unitary, {qubit: 0}, cycle_check=False)
        return dag

    def _check_not_in_basis(self, gate_name, qargs, global_index_map):
        if self.target is not None:
            return not self.target.instruction_supported(
                gate_name, tuple(global_index_map[qubit] for qubit in qargs)
            )
        else:
            return self.basis_gates and gate_name not in self.basis_gates

    def _block_qargs_to_indices(self, block_qargs, global_index_map):
        """Map each qubit in block_qargs to its wire position among the block's wires.
        Args:
            block_qargs (list): list of qubits that a block acts on
            global_index_map (dict): mapping from each qubit in the
                circuit to its wire position within that circuit
        Returns:
            dict: mapping from qarg to position in block
        """
        block_indices = [global_index_map[q] for q in block_qargs]
        ordered_block_indices = {bit: index for index, bit in enumerate(sorted(block_indices))}
        block_positions = {q: ordered_block_indices[global_index_map[q]] for q in block_qargs}
        return block_positions
