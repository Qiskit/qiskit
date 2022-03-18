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

"""Rearrange the direction of the cx nodes to match the directed coupling map."""

from math import pi

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import RYGate, HGate, CXGate, ECRGate, RZXGate


class GateDirection(TransformationPass):
    """Modify asymmetric gates to match the hardware coupling direction.

    This pass makes use of the following identities::

                             ┌───┐┌───┐┌───┐
        q_0: ──■──      q_0: ┤ H ├┤ X ├┤ H ├
             ┌─┴─┐  =        ├───┤└─┬─┘├───┤
        q_1: ┤ X ├      q_1: ┤ H ├──■──┤ H ├
             └───┘           └───┘     └───┘

             ┌──────┐          ┌───────────┐┌──────┐┌───┐
        q_0: ┤0     ├     q_0: ┤ RY(-pi/2) ├┤1     ├┤ H ├
             │  ECR │  =       └┬──────────┤│  ECR │├───┤
        q_1: ┤1     ├     q_1: ─┤ RY(pi/2) ├┤0     ├┤ H ├
             └──────┘           └──────────┘└──────┘└───┘

             ┌──────┐          ┌───┐┌──────┐┌───┐
        q_0: ┤0     ├     q_0: ┤ H ├┤1     ├┤ H ├
             │  RZX │  =       ├───┤│  RZX │├───┤
        q_1: ┤1     ├     q_1: ┤ H ├┤0     ├┤ H ├
             └──────┘          └───┘└──────┘└───┘
    """

    def __init__(self, coupling_map, target=None):
        """GateDirection pass.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            target (Target): The backend target to use for this pass. If this is specified
                it will be used instead of the coupling map
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target

        # Create the replacement dag and associated register.
        self._cx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        self._cx_dag.add_qreg(qr)
        self._cx_dag.apply_operation_back(HGate(), [qr[0]], [])
        self._cx_dag.apply_operation_back(HGate(), [qr[1]], [])
        self._cx_dag.apply_operation_back(CXGate(), [qr[1], qr[0]], [])
        self._cx_dag.apply_operation_back(HGate(), [qr[0]], [])
        self._cx_dag.apply_operation_back(HGate(), [qr[1]], [])

        self._ecr_dag = DAGCircuit()
        qr = QuantumRegister(2)
        self._ecr_dag.add_qreg(qr)
        self._ecr_dag.apply_operation_back(RYGate(-pi / 2), [qr[0]], [])
        self._ecr_dag.apply_operation_back(RYGate(pi / 2), [qr[1]], [])
        self._ecr_dag.apply_operation_back(ECRGate(), [qr[1], qr[0]], [])
        self._ecr_dag.apply_operation_back(HGate(), [qr[0]], [])
        self._ecr_dag.apply_operation_back(HGate(), [qr[1]], [])

    @staticmethod
    def _rzx_dag(parameter):
        _rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        _rzx_dag.add_qreg(qr)
        _rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        _rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        _rzx_dag.apply_operation_back(RZXGate(parameter), [qr[1], qr[0]], [])
        _rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        _rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        return _rzx_dag

    def run(self, dag):
        """Run the GateDirection pass on `dag`.

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
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        layout_map = trivial_layout.get_virtual_bits()
        if len(dag.qregs) > 1:
            raise TranspilerError(
                "GateDirection expects a single qreg input DAG,"
                "but input DAG had qregs: {}.".format(dag.qregs)
            )
        if self.target is None:
            cmap_edges = set(self.coupling_map.get_edges())
            if not cmap_edges:
                return dag

            self.coupling_map.compute_distance_matrix()

            dist_matrix = self.coupling_map.distance_matrix

            for node in dag.two_qubit_ops():
                control = node.qargs[0]
                target = node.qargs[1]

                physical_q0 = layout_map[control]
                physical_q1 = layout_map[target]

                if dist_matrix[physical_q0, physical_q1] != 1:
                    raise TranspilerError(
                        "The circuit requires a connection between physical "
                        "qubits %s and %s" % (physical_q0, physical_q1)
                    )

                if (physical_q0, physical_q1) not in cmap_edges:
                    if node.name == "cx":
                        dag.substitute_node_with_dag(node, self._cx_dag)
                    elif node.name == "ecr":
                        dag.substitute_node_with_dag(node, self._ecr_dag)
                    elif node.name == "rzx":
                        dag.substitute_node_with_dag(node, self._rzx_dag(*node.op.params))
                    else:
                        raise TranspilerError(
                            f"Flipping of gate direction is only supported "
                            f"for CX, ECR, and RZX at this time, not {node.name}."
                        )
        else:
            # TODO: Work with the gate instances and only use names as look up keys.
            # This will require iterating over the target names to build a mapping
            # of names to gates that implement CXGate, ECRGate, RZXGate (including
            # fixed angle variants)
            for node in dag.two_qubit_ops():
                control = node.qargs[0]
                target = node.qargs[1]

                physical_q0 = layout_map[control]
                physical_q1 = layout_map[target]

                if node.name == "cx":
                    if (physical_q0, physical_q1) in self.target["cx"]:
                        continue
                    if (physical_q1, physical_q0) in self.target["cx"]:
                        dag.substitute_node_with_dag(node, self._cx_dag)
                    else:
                        raise TranspilerError(
                            "The circuit requires a connection between physical "
                            "qubits %s and %s for cx" % (physical_q0, physical_q1)
                        )
                elif node.name == "ecr":
                    if (physical_q0, physical_q1) in self.target["ecr"]:
                        continue
                    if (physical_q1, physical_q0) in self.target["ecr"]:
                        dag.substitute_node_with_dag(node, self._ecr_dag)
                    else:
                        raise TranspilerError(
                            "The circuit requires a connection between physical "
                            "qubits %s and %s for ecr" % (physical_q0, physical_q1)
                        )
                elif node.name == "rzx":
                    if (physical_q0, physical_q1) in self.target["rzx"]:
                        continue
                    if (physical_q1, physical_q0) in self.target["rzx"]:
                        dag.substitute_node_with_dag(node, self._rzx_dag(*node.op.params))
                    else:
                        raise TranspilerError(
                            "The circuit requires a connection between physical "
                            "qubits %s and %s for rzx" % (physical_q0, physical_q1)
                        )
                else:
                    raise TranspilerError(
                        f"Flipping of gate direction is only supported "
                        f"for CX, ECR, and RZX at this time, not {node.name}."
                    )
        return dag
