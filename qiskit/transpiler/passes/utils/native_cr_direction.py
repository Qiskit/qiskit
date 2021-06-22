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

"""Rearrange the direction of the cx nodes to match the hardware-native backend cx direction."""

from math import pi
from typing import Tuple

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import RYGate, HGate, CXGate, ECRGate, RZXGate


class NativeCRGateDirection(TransformationPass):
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
             ┌─────────┐            ┌───┐┌─────────┐┌───┐
        q_0: ┤0        ├       q_0: ┤ H ├┤1        ├┤ H ├
             │  Rzx(θ) │  =         ├───┤│  Rzx(θ) │├───┤
        q_1: ┤1        ├       q_1: ┤ H ├┤0        ├┤ H ├
             └─────────┘            └───┘└─────────┘└───┘
    """

    def __init__(self, backend):
        """GateDirection pass.
        Args:
            backend (BaseBackend): The hardware backend that is used.
        """
        super().__init__()
        self.backend = backend

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

    def _rzx_dag(self, theta):

        rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        rzx_dag.add_qreg(qr)
        rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        rzx_dag.apply_operation_back(RZXGate(theta), [qr[1], qr[0]], [])
        rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        return rzx_dag

    def is_native_cx(self, qubit_pair: Tuple):
        """Check that a CX for a qubit pair is native."""
        inst_map = self.backend.defaults().instruction_schedule_map
        cx1 = inst_map.get('cx', qubit_pair)
        cx2 = inst_map.get('cx', qubit_pair[::-1])
        return cx1.duration < cx2.duration

    def run(self, dag):
        """Run the GateDirection pass on `dag`.
        Flips the cx, ecr, and rzx nodes to match the hardware-native coupling map.
        Modifies the input dag.
        Args:
            dag (DAGCircuit): DAG to map.
        Returns:
            DAGCircuit: The rearranged dag for the hardware-native coupling map.
        Raises:
            TranspilerError: If the circuit cannot be mapped just by flipping the
                cx nodes.
        """

        if len(dag.qregs) > 1:
            raise TranspilerError('GateDirection expects a single qreg input DAG,'
                                  'but input DAG had qregs: {}.'.format(
                dag.qregs))

        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        for node in dag.two_qubit_ops():
            control = node.qargs[0]
            target = node.qargs[1]

            physical_q0 = trivial_layout[control]
            physical_q1 = trivial_layout[target]

            config = self.backend.configuration()
            if [physical_q0, physical_q1] not in config.coupling_map:
                raise TranspilerError('Qubits %s and %s are not connected on the backend'
                                      % (physical_q0, physical_q1))

            if not self.is_native_cx((physical_q0, physical_q1)):
                if node.name == 'cx':
                    dag.substitute_node_with_dag(node, self._cx_dag)
                elif node.name == 'ecr':
                    dag.substitute_node_with_dag(node, self._ecr_dag)
                elif node.name == 'rzx':
                    theta = node.op.params[0]
                    dag.substitute_node_with_dag(node, self._rzx_dag(theta))
                else:
                    raise TranspilerError('Flipping of gate direction is only supported '
                                          'for CX and ECR at this time.')

        return dag
