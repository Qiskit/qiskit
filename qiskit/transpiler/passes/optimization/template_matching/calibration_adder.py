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

"""Scheduling transformation pass to add calibrations to the dag."""

from qiskit.transpiler.passes.scheduling import CalibrationCreator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import Gate
from qiskit.extensions import UnitaryGate


class CalibrationAdder(TransformationPass):
    """Identifies gates for which we can add calibrations."""

    def __init__(self, calibration_creator: CalibrationCreator):
        """
        Args:
            calibration_creator: An instance of CalibrationCreator capable of generating the
                schedules that will be added to the circuit.
        """
        super().__init__()
        self._calibration_adder = calibration_creator

    def run(self, dag):
        """Run the calibration adder pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A DAG with calibrations added to it.
        """
        for node in dag.nodes():
            if self._calibration_adder.supported(node.name):
                name = node.name
                qubits = [_.index for _ in node.qargs]

                schedule, params = self._calibration_adder.get_calibration(name, qubits)

                dag.add_calibration(name, qubits, schedule, params=params)

                # Unitary gate has unhashable data
                if isinstance(node.op, UnitaryGate):
                    node.op = Gate(name, node.op.num_qubits, params)

        return dag
