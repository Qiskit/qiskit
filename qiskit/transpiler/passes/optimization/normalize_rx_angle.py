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
Quantize the RX rotation angles by assigning the same value for the angles
that differ within a resolution provided by the user.
"""

import numpy as np

from qiskit.utils import deprecate_func

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import RXGate, RZGate, SXGate, XGate


class NormalizeRXAngle(TransformationPass):
    """Normalize theta parameter of RXGate instruction.

    The parameter normalization is performed with following steps.

    1) Wrap RX Gate theta into [0, pi]. When theta is negative value, the gate is
    decomposed into the following sequence.

    .. code-block::

          ┌───────┐┌─────────┐┌────────┐
       q: ┤ Rz(π) ├┤ Rx(|θ|) ├┤ Rz(-π) ├
          └───────┘└─────────┘└────────┘

    2) If the operation is supported by target, convert RX(pi/2) to SX, and RX(pi) to X.

    3) Quantize theta value according to the user-specified resolution.

    This will help reduce the size of calibration data sent over the wire,
    and allow us to exploit the more accurate, hardware-calibrated pulses.
    Note that pulse calibration might be attached per each rotation angle.
    """

    @deprecate_func(
        since="1.4",
        removal_timeline="in Qiskit 2.0",
        additional_msg="This pass was used as pre-processing step of ``RXCalibrationBuilder``."
        " With the removal of Pulse in Qiskit 2.0, this pass is no longer needed.",
    )
    def __init__(self, target=None, resolution_in_radian=0):
        """NormalizeRXAngle initializer.

        Args:
            target (Target): The :class:`~.Target` representing the target backend.
                If the target contains SX and X calibrations, this pass will replace the
                corresponding RX gates with SX and X gates.
            resolution_in_radian (float): Resolution for RX rotation angle quantization.
                If set to zero, this pass won't modify the rotation angles in the given DAG.
                (=Provides arbitrary-angle RX)
        """
        super().__init__()
        self.target = target
        self.resolution_in_radian = resolution_in_radian
        self.already_generated = {}

    def quantize_angles(self, qubit, original_angle):
        """Quantize the RX rotation angles by assigning the same value for the angles
        that differ within a resolution provided by the user.

        Args:
            qubit (qiskit.circuit.Qubit): This will be the dict key to access the list of
                quantized rotation angles.
            original_angle (float): Original rotation angle, before quantization.

        Returns:
            float: Quantized angle.
        """

        if (angles := self.already_generated.get(qubit)) is None:
            self.already_generated[qubit] = np.array([original_angle])
            return original_angle
        similar_angles = angles[
            np.isclose(angles, original_angle, atol=self.resolution_in_radian / 2)
        ]
        if similar_angles.size == 0:
            self.already_generated[qubit] = np.append(angles, original_angle)
            return original_angle
        return float(similar_angles[0])

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

            half_pi_rotation = np.isclose(
                abs(wrapped_theta), np.pi / 2, atol=self.resolution_in_radian / 2
            )
            pi_rotation = np.isclose(abs(wrapped_theta), np.pi, atol=self.resolution_in_radian / 2)

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
