"""Quantize RX angles with a resolution provided by the user"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister
from qiskit.circuit.library.standard_gates import RXGate, RZGate, SXGate, XGate

import numpy as np


class QuantizeRXAngle(TransformationPass):
    """Quantizes the rotation angles for RX gates.
    """

    def __init__(self, target=None, resolution_in_radian=0):
        """QuantizeRXAngle initializer.

        Args:
            target (Target): The :class:`~.Target` representing the target backend.
                If the target contains SX and X calibrations, this pass will replace the
                corresponding RX gates with SX and X gates.
            resolution_in_radian (float): Resolution for RX rotation angle quantization.
            If set to zero, the pass doesn't change anything. (=Provides aribitary-angle RX)
        """
        super().__init__()
        self.target = target
        self.resolution_in_radian = resolution_in_radian
        self.already_generated = {}

    def run(self, dag):
        """Run the QuantizeRXAngle pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the DAG where all RX rotation angles are within [0, pi].
        """
        # Don't perform quantization if resolution is zero
        if self.resolution_in_radian == 0:
            return dag
        
        # Iterate over all op_nodes and quantize RX angles
        for op_node in dag.op_nodes():
            if not (op_node.op.name == "rx"):
                continue

            original_angle = op_node.op.params[0]
            qubit = dag.find_bit(op_node.qargs[0]).index

            # check if there is already a calibration for a simliar angle
            try:
                angles = self.already_generated[qubit]  # 1d ndarray of already generated angles
                angle = float(
                    angles[np.where(np.abs(angles - original_angle) < (self.resolution_in_radian / 2))]
                )
            except KeyError:
                # there's no calibration at all for the given qubit index
                quantized_angle = original_angle
                self.already_generated[qubit] = np.array([angle])
            except TypeError:
                # TypeError happens when typecasting to float.
                # It means that there's no calibration for this angle
                quantized_angle = original_angle
                self.already_generated[qubit] = np.append(self.already_generated[qubit], angle)
            
            op_node.op.params[0] = quantized_angle
            
        return dag
