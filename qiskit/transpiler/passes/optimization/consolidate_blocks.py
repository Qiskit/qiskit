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
from __future__ import annotations

import numpy as np

from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library.standard_gates import CXGate, CZGate, iSwapGate, ECRGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.synthesis import unitary_synthesis
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit._accelerate.consolidate_blocks import consolidate_blocks
from qiskit.exceptions import QiskitError

from .collect_1q_runs import Collect1qRuns
from .collect_2q_blocks import Collect2qBlocks

KAK_GATE_NAMES = {
    "cx": CXGate(),
    "cz": CZGate(),
    "iswap": iSwapGate(),
    "ecr": ECRGate(),
}


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

    def __init__(
        self,
        kak_basis_gate=None,
        force_consolidate=False,
        basis_gates=None,
        approximation_degree=1.0,
        target=None,
    ):
        """ConsolidateBlocks initializer.

        If ``kak_basis_gate`` is not ``None`` it will be used as the basis gate for KAK decomposition.
        Otherwise, if ``basis_gates`` is not ``None`` a basis gate will be chosen from this list.
        Otherwise, the basis gate will be :class:`.CXGate`.

        Args:
            kak_basis_gate (Gate): Basis gate for KAK decomposition.
            force_consolidate (bool): Force block consolidation.
            basis_gates (List(str)): Basis gates from which to choose a KAK gate.
            approximation_degree (float): a float between :math:`[0.0, 1.0]`. Lower approximates more.
            target (Target): The target object for the compilation target backend.
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
            kak_gates = set(basis_gates or []).intersection(KAK_GATE_NAMES.keys())
            kak_gate = None
            if kak_gates:
                kak_gate = KAK_GATE_NAMES[kak_gates.pop()]
                self.decomposer = TwoQubitBasisDecomposer(
                    kak_gate, basis_fidelity=approximation_degree or 1.0
                )
            if kak_gate is None:
                self.decomposer = TwoQubitBasisDecomposer(CXGate())
        else:
            self.decomposer = TwoQubitBasisDecomposer(CXGate())

    def run(self, dag):
        """Run the ConsolidateBlocks pass on `dag`.

        Iterate over each block and replace it with an equivalent Unitary
        on the same wires.
        """
        if self.decomposer is None:
            return dag

        blocks = self.property_set["block_list"]
        if blocks is not None:
            blocks = [[node._node_id for node in block] for block in blocks]
        runs = self.property_set["run_list"]
        if runs is not None:
            runs = [[node._node_id for node in run] for run in runs]

        consolidate_blocks(
            dag,
            self.decomposer._inner_decomposer,
            self.force_consolidate,
            target=self.target,
            basis_gates=self.basis_gates,
            blocks=blocks,
            runs=runs,
        )
        dag = self._handle_control_flow_ops(dag)

        # Clear collected blocks and runs as they are no longer valid after consolidation
        if "run_list" in self.property_set:
            del self.property_set["run_list"]
        if "block_list" in self.property_set:
            del self.property_set["block_list"]

        return dag

    def _handle_control_flow_ops(self, dag):
        """
        This is similar to transpiler/passes/utils/control_flow.py except that the
        collect blocks is redone for the control flow blocks.
        """

        pass_manager = PassManager()
        if "run_list" in self.property_set:
            pass_manager.append(Collect1qRuns())
        if "block_list" in self.property_set:
            pass_manager.append(Collect2qBlocks())

        pass_manager.append(self)
        for node in dag.op_nodes():
            if node.name not in CONTROL_FLOW_OP_NAMES:
                continue
            dag.substitute_node(
                node,
                node.op.replace_blocks(pass_manager.run(block) for block in node.op.blocks),
                propagate_condition=False,
            )
        return dag

    def _check_not_in_basis(self, dag, gate_name, qargs):
        if self.target is not None:
            return not self.target.instruction_supported(
                gate_name, tuple(dag.find_bit(qubit).index for qubit in qargs)
            )
        else:
            return self.basis_gates and gate_name not in self.basis_gates

    def _block_qargs_to_indices(self, dag, block_qargs):
        """Map each qubit in block_qargs to its wire position among the block's wires.
        Args:
            block_qargs (list): list of qubits that a block acts on
            global_index_map (dict): mapping from each qubit in the
                circuit to its wire position within that circuit
        Returns:
            dict: mapping from qarg to position in block
        """
        block_indices = [dag.find_bit(q).index for q in block_qargs]
        ordered_block_indices = {bit: index for index, bit in enumerate(sorted(block_indices))}
        block_positions = {q: ordered_block_indices[dag.find_bit(q).index] for q in block_qargs}
        return block_positions
