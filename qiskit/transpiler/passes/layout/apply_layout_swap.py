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

"""TODO"""
from typing import Union

import numpy as np

from qiskit.transpiler import Layout, CouplingMap

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.layout_transformation import LayoutTransformation
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper


class ApplyLayoutSwaps(TransformationPass):
    """TODO"""
    def __init__(self, coupling_map: CouplingMap,
                 seed: Union[int, np.random.default_rng] = None,
                 trials: int = 4):
        super().__init__()
        self.coupling_map = coupling_map
        self.seed = seed
        self.trials = trials

    def run(self, dag):
        """Run the ApplyLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG (with physical qubits).

        Raises:
            TranspilerError: if no layout is found in `property_set` or no full physical qubits.
        """
        layout = self.property_set["layout"]
        if not layout:
            raise TranspilerError(
                "No 'layout' is found in property_set. Please run a Layout pass in advance.")
        if len(layout) != (1 + max(layout.get_physical_bits())):
            raise TranspilerError(
                "The 'layout' must be full (with ancilla).")

        if self.coupling_map:
            graph = self.coupling_map.graph.to_undirected()
        else:
            coupling_map = CouplingMap.from_full(len(layout))
            graph = coupling_map.graph

        token_swapper = ApproximateTokenSwapper(graph, self.seed)

        q = QuantumRegister(len(layout), 'q')

        new_dag = DAGCircuit()
        new_dag.add_qreg(q)
        new_dag.metadata = dag.metadata
        new_dag._global_phase = dag._global_phase
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        # Find the permutation from trivial layout to property_set['layout'].
        permutation = {pqubit: layout.get_virtual_bits()[vqubit]
                       for vqubit, pqubit in trivial_layout.get_virtual_bits().items()}
        perm_circ = token_swapper.permutation_circuit(permutation, self.trials)

        qubits = [dag.qubits[i[0]] for i in sorted(perm_circ.inputmap.items(), key=lambda x: x[0])]
        new_dag.compose(perm_circ.circuit, qubits=qubits)

        for node in dag.topological_op_nodes():
            if node.type == 'op':
                qargs = [q[layout[qarg]] for qarg in node.qargs]
                new_dag.apply_operation_back(node.op, qargs, node.cargs)

        return new_dag
