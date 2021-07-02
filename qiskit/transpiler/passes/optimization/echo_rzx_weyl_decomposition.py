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

"""Weyl decomposition of two-qubit gates in terms of echoed cross-resonance gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag

from qiskit.transpiler.layout import Layout

import qiskit.quantum_info as qi

from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylEchoRZX


class EchoRZXWeylDecomposition(TransformationPass):
    """Rewrite two-qubit gates using the Weyl decomposition.

    This transpiler pass rewrites two-qubit gates in terms of echoed cross-resonance gates according
    to the Weyl decomposition. A two-qubit gate will be replaced with at most six non-echoed RZXGates.
    Each pair of RZXGates forms an echoed RZXGate.
    """

    def __init__(self, inst_map):
        """EchoRZXWeylDecomposition pass."""
        self.inst_map = inst_map        
        super().__init__()

    def run(self, dag):
        """Run the EchoRZXWeylDecomposition pass on `dag`.

        Rewrites two-qubit gates in an arbitrary circuit in terms of echoed cross-resonance
        gates by computing the Weyl decomposition of the corresponding unitary. Modifies the
        input dag.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: The rearranged dag.

        Raises:
            TranspilerError: If the circuit cannot be mapped.
        """

        if len(dag.qregs) > 1:
            raise TranspilerError(
                "EchoRZXWeylDecomposition expects a single qreg input DAG,"
                f"but input DAG had qregs: {dag.qregs}."
            )

        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        for node in dag.two_qubit_ops():
            if node.type == "op":
                control = node.qargs[0]
                target = node.qargs[1]

                physical_q0 = trivial_layout[control]
                physical_q1 = trivial_layout[target]

                qubit_pair = (physical_q0, physical_q1)

                unitary = qi.Operator(node.op).data
                dag_weyl = circuit_to_dag(
                    TwoQubitWeylEchoRZX(
                        unitary, inst_map=self.inst_map, qubit_pair=qubit_pair
                    ).circuit()
                )
                dag.substitute_node_with_dag(node, dag_weyl)

        return dag
