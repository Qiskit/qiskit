# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Performs three optimizations to reduce the number of pulse calibrations for
the single-pulse RX gates:
Wrap RX Gate rotation angles into [0, pi] by sandwiching them with RZ gates.
Convert RX(pi/2) to SX, and RX(pi) to X if the calibrations exist in the target.
Quantize the RX rotation angles using a resolution provided by the user.
"""

import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import RXGate, RZGate, SXGate, XGate


class NormalizeRXAngle(TransformationPass):
    """Wrap RX Gate rotation angles into [0, pi] by sandwiching them with RZ gates.
    This will help reduce the size of calibration data,
    as we won't have to keep separate, phase-flipped calibrations for negative rotation angles.
    Moreover, if the calibrations exist in the target, convert RX(pi/2) to SX, and RX(pi) to X.
    This will allow us to exploit the more accurate, hardware-calibrated pulses.
    Lastly, quantize the RX rotation angles using a resolution provided by the user.
    """

    def __init__(self, target=None, resolution_in_radian=0):
        """NormalizeRXAngle initializer.

        Args:
            target (Target): The :class:`~.Target` representing the target backend.
                If the target contains SX and X calibrations, this pass will replace the
                corresponding RX gates with SX and X gates.
            resolution_in_radian (float): Resolution for RX rotation angle quantization.
                If set to zero, this pass won't modify the rotation angles in the given DAG.
                (=Provides aribitary-angle RX)
        """
        super().__init__()
        self.target = target
        self.resolution_in_radian = resolution_in_radian
        self.already_generated = {}

    def quantize_angles(self, qubit, original_angle):
        """Quantize the RX rotation angles using a resolution provided by the user.

        Args:
            qubit (Qubit): This will be the dict key to access the list of quantized rotation angles.
            original_angle (float): Original rotation angle, before quantization.

        Returns:
            float: Quantized angle.
        """

        # check if there is already a calibration for a simliar angle
        try:
            angles = self.already_generated[qubit]  # 1d ndarray of already generated angles
            similar_angle = angles[
                np.isclose(angles, original_angle, atol=self.resolution_in_radian / 2)
            ]
            quantized_angle = (
                float(similar_angle[0]) if len(similar_angle) > 1 else float(similar_angle)
            )
        except KeyError:
            quantized_angle = original_angle
            self.already_generated[qubit] = np.array([quantized_angle])
        except TypeError:
            quantized_angle = original_angle
            self.already_generated[qubit] = np.append(
                self.already_generated[qubit], quantized_angle
            )

        return quantized_angle

    def run(self, dag):
        """Run the NormalizeRXAngle pass on ``dag``.

        Args:
            dag (DAGCircuit): The DAG to be optimized.

        Returns:
            DAGCircuit: A DAG with RX gate calibration.
        """

        # Iterate over all op_nodes and replace RX if eligible for modification.
        for op_node in dag.op_nodes():
            if not isinstance(op_node.op, RXGate):
                continue

            raw_theta = op_node.op.params[0]
            wrapped_theta = np.arctan2(np.sin(raw_theta), np.cos(raw_theta))  # [-pi, pi]

            if self.resolution_in_radian:
                wrapped_theta = self.quantize_angles(op_node.qargs[0], wrapped_theta)

            half_pi_rotation = np.isclose(abs(wrapped_theta), np.pi / 2)
            pi_rotation = np.isclose(abs(wrapped_theta), np.pi)

            should_modify_node = (
                (wrapped_theta != raw_theta)
                or (wrapped_theta < 0)
                or half_pi_rotation
                or pi_rotation
            )

            if should_modify_node:
                mini_dag = DAGCircuit()
                mini_dag.add_qubits(op_node.qargs)

                # new X-rotation gate with angle in [0, pi]
                if half_pi_rotation:
                    physical_qubit_idx = dag.find_bit(op_node.qargs[0]).index
                    if self.target.instruction_supported("sx", (physical_qubit_idx,)):
                        mini_dag.apply_operation_back(SXGate(), qargs=op_node.qargs)
                elif pi_rotation:
                    physical_qubit_idx = dag.find_bit(op_node.qargs[0]).index
                    if self.target.instruction_supported("x", (physical_qubit_idx,)):
                        mini_dag.apply_operation_back(XGate(), qargs=op_node.qargs)
                else:
                    mini_dag.apply_operation_back(
                        RXGate(np.abs(wrapped_theta)), qargs=op_node.qargs
                    )

                # sandwich with RZ if the intended rotation angle was negative
                if wrapped_theta < 0:
                    mini_dag.apply_operation_front(RZGate(np.pi), qargs=op_node.qargs)
                    mini_dag.apply_operation_back(RZGate(-np.pi), qargs=op_node.qargs)

                dag.substitute_node_with_dag(node=op_node, input_dag=mini_dag, wires=op_node.qargs)

        return dag
