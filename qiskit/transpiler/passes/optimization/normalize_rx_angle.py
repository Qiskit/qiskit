"""Wrap RX Gate rotation angles into [0, pi] by sandwiching them with RZ gates.
Convert RX(pi/2) to SX, and RX(pi) to X if the calibrations exist in the target.
"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister
from qiskit.circuit.library.standard_gates import RXGate, RZGate, SXGate, XGate

import numpy as np


class NormalizeRXAngle(TransformationPass):
    """Wrap RX Gate rotation angles into [0, pi] by sandwiching them with RZ gates.
    This will help reducing the size of calibration data,
    as we don't have to keep separate, phase-flipped calibrations for negative rotation angles.
    Moreover, convert RX(pi/2) to SX, and RX(pi) to X 
    if the calibrations exist in the target.
    This will let us exploit the hardware-calibrated pulses.
    """

    def __init__(self, target=None, resolution_in_radian=0):
        """NormalizeRXAngle initializer.

        Args:
            target (Target): The :class:`~.Target` representing the target backend.
                If the target contains SX and X calibrations, this pass will replace the
                corresponding RX gates with SX and X gates.
        """
        super().__init__()
        self.target = target

    def run(self, dag):
        """Run the NormalizeRXAngle pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the DAG where all RX rotation angles are within [0, pi].
        """

        # Iterate over all op_nodes and replace RX if eligible for modification.
        for op_node in dag.op_nodes():
            if not (op_node.op.name == "rx"):
                continue

            raw_theta = op_node.op.params[0]
            wrapped_theta = np.arctan2(np.sin(raw_theta), np.cos(raw_theta))  # [-pi, pi]

            half_pi_rotation = np.isclose(abs(wrapped_theta), np.pi / 2)
            pi_rotation = np.isclose(abs(wrapped_theta), np.pi)

            # get the physical qubit index to look up the SX or X calibrations
            qubit = dag.find_bit(op_node.qargs[0]).index if half_pi_rotation | pi_rotation else None
            try:
                qubit = int(qubit)
                find_bit_succeeded = True
            except TypeError as e:
                find_bit_succeeded = False

            should_modify_node = (
                (wrapped_theta != raw_theta)
                or (wrapped_theta < 0)
                or half_pi_rotation
                or pi_rotation
            )

            if should_modify_node:
                mini_dag = DAGCircuit()
                temp_qreg = QuantumRegister(1, "temp_qreg")
                mini_dag.add_qreg(temp_qreg)

                # new X-rotation gate with angle in [0, pi]
                if half_pi_rotation and find_bit_succeeded and self.target.has_calibration("sx", (qubit,)):
                    mini_dag.apply_operation_back(SXGate(), qargs=temp_qreg)
                elif pi_rotation and find_bit_succeeded and self.target.has_calibration("x", (qubit,)):
                    mini_dag.apply_operation_back(XGate(), qargs=temp_qreg)
                else:
                    mini_dag.apply_operation_back(RXGate(np.abs(wrapped_theta)), qargs=temp_qreg)

                # sandwich with RZ if the intended rotation angle was negative
                if wrapped_theta < 0:
                    mini_dag.apply_operation_front(RZGate(np.pi), qargs=temp_qreg)
                    mini_dag.apply_operation_back(RZGate(-np.pi), qargs=temp_qreg)

                dag.substitute_node_with_dag(node=op_node, input_dag=mini_dag, wires=temp_qreg)

        return dag
