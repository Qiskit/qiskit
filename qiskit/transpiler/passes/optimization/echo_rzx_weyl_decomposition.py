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

import qiskit.quantum_info as qi

from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylEchoRZX


class EchoRZXWeylDecomposition(TransformationPass):
    """Rewrite two-qubit gates in terms of echoed cross-resonance gates according
    to the Weyl decomposition.
    """

    def __init__(self):
        """EchoRZXWeylDecomposition pass.
        """
        super().__init__()

    def run(self, dag):
        """Run the EchoRZXWeylDecomposition pass on `dag`.
        Rewrites two-qubit gates in an arbitrary circuit in terms of echoed cross-resonance
        gates by computing the Cartan decomposition of the corresponding unitary. Modifies the
        input dag.
        Args:
            dag (DAGCircuit): DAG to map.
        Returns:
            DAGCircuit: The rearranged dag.
        Raises:
            TranspilerError: If the circuit cannot be mapped.
        """

        if len(dag.qregs) > 1:
            raise TranspilerError('EchoRZXWeylDecomposition expects a single qreg input DAG,'
                                  'but input DAG had qregs: {}.'.format(dag.qregs))

        for idx, node in enumerate(dag.two_qubit_ops()):
            if node.type == "op":
                unitary = qi.Operator(node.op).data
                dag_weyl = circuit_to_dag(TwoQubitWeylEchoRZX(unitary).circuit())
                dag.substitute_node_with_dag(node, dag_weyl)

        return dag
