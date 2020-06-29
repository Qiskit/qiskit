# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rearrange the direction of the cx nodes to match the directed coupling map."""

from math import pi

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import U2Gate, CXGate, ECRGate


class CXDirection(TransformationPass):
    """Rearrange the direction of the cx nodes to match the directed coupling map.

    Flip the cx nodes to match the directed coupling map. This pass makes use
    of the following equivalence::

        ---(+)---      --[H]---.---[H]--
            |      =           |
        ----.----      --[H]--(+)--[H]--
    """

    def __init__(self, coupling_map):
        """CXDirection initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        """Run the CXDirection pass on `dag`.

        Flips the cx nodes to match the directed coupling map. Modifies the
        input dag.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: The rearranged dag for the coupling map

        Raises:
            TranspilerError: If the circuit cannot be mapped just by flipping the
                cx nodes.
        """
        cmap_edges = set(self.coupling_map.get_edges())

        if len(dag.qregs) > 1:
            raise TranspilerError('CXDirection expects a single qreg input DAG,'
                                  'but input DAG had qregs: {}.'.format(
                                      dag.qregs))

        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        for node in dag.two_qubit_ops():
            control = node.qargs[0]
            target = node.qargs[1]

            physical_q0 = trivial_layout[control]
            physical_q1 = trivial_layout[target]

            if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                raise TranspilerError('The circuit requires a connection between physical '
                                      'qubits %s and %s' % (physical_q0, physical_q1))

            if (physical_q0, physical_q1) not in cmap_edges:
                # Create the replacement dag and associated register.
                sub_dag = DAGCircuit()
                sub_qr = QuantumRegister(2)
                sub_dag.add_qreg(sub_qr)

                if node.name == 'cx':
                    """                  ┌───┐┌───┐┌───┐
                    q_0: ──■──      q_0: ┤ H ├┤ X ├┤ H ├
                         ┌─┴─┐  =        ├───┤└─┬─┘├───┤
                    q_1: ┤ X ├      q_1: ┤ H ├──■──┤ H ├
                         └───┘           └───┘     └───┘
                    """
                    sub_dag.apply_operation_back(U2Gate(0, pi), [sub_qr[0]], [])
                    sub_dag.apply_operation_back(U2Gate(0, pi), [sub_qr[1]], [])
                    sub_dag.apply_operation_back(CXGate(), [sub_qr[1], sub_qr[0]], [])
                    sub_dag.apply_operation_back(U2Gate(0, pi), [sub_qr[0]], [])
                    sub_dag.apply_operation_back(U2Gate(0, pi), [sub_qr[1]], [])
                elif node.name == 'ecr':
                    """
                         ┌──────┐          ┌───────────┐┌──────┐┌───┐
                    q_0: ┤0     ├     q_0: ┤ RY(-pi/2) ├┤1     ├┤ H ├
                         │  ECR │  =       └┬──────────┤│  ECR │├───┤
                    q_1: ┤1     ├     q_1: ─┤ RY(pi/2) ├┤0     ├┤ H ├
                         └──────┘           └──────────┘└──────┘└───┘
                    """
                    sub_dag.apply_operation_back(U2Gate(pi, pi), [sub_qr[0]], [])
                    sub_dag.apply_operation_back(U2Gate(0, 0), [sub_qr[1]], [])
                    sub_dag.apply_operation_back(ECRGate(), [sub_qr[1], sub_qr[0]], [])
                    sub_dag.apply_operation_back(U2Gate(0, pi), [sub_qr[0]], [])
                    sub_dag.apply_operation_back(U2Gate(0, pi), [sub_qr[1]], [])

                dag.substitute_node_with_dag(node, sub_dag)

        return dag
