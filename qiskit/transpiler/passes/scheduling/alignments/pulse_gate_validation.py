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

"""Analysis passes for hardware alignment constraints."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import Play
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class ValidatePulseGates(AnalysisPass):
    """Check custom gate length.

    This is a control electronics aware analysis pass.

    Quantum gates (instructions) are often implemented with shaped analog stimulus signals.
    These signals may be digitally stored in the waveform memory of the control electronics
    and converted into analog voltage signals by electronic components known as
    digital to analog converters (DAC).

    In Qiskit SDK, we can define the pulse-level implementation of custom quantum gate
    instructions, as a `pulse gate
    <https://qiskit.org/documentation/tutorials/circuits_advanced/05_pulse_gates.html>`__,
    thus user gates should satisfy all waveform memory constraints imposed by the backend.

    This pass validates all attached calibration entries and raises ``TranspilerError`` to
    kill the transpilation process if any invalid calibration entry is found.
    This pass saves users from waiting until job execution time to get an invalid pulse error from
    the backend control electronics.
    """

    def __init__(
        self,
        granularity: int = 1,
        min_length: int = 1,
    ):
        """Create new pass.

        Args:
            granularity: Integer number representing the minimum time resolution to
                define the pulse gate length in units of ``dt``. This value depends on
                the control electronics of your quantum processor.
            min_length: Integer number representing the minimum data point length to
                define the pulse gate in units of ``dt``. This value depends on
                the control electronics of your quantum processor.
        """
        super().__init__()
        self.granularity = granularity
        self.min_length = min_length

    def run(self, dag: DAGCircuit):
        """Run the pulse gate validation attached to ``dag``.

        Args:
            dag: DAG to be validated.

        Returns:
            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.

        Raises:
            TranspilerError: When pulse gate violate pulse controller constraints.
        """
        if self.granularity == 1 and self.min_length == 1:
            # we can define arbitrary length pulse with dt resolution
            return

        for gate, insts in dag.calibrations.items():
            for qubit_param_pair, schedule in insts.items():
                for _, inst in schedule.instructions:
                    if isinstance(inst, Play):
                        pulse = inst.pulse
                        if pulse.duration % self.granularity != 0:
                            raise TranspilerError(
                                f"Pulse duration is not multiple of {self.granularity}. "
                                "This pulse cannot be played on the specified backend. "
                                f"Please modify the duration of the custom gate pulse {pulse.name} "
                                f"which is associated with the gate {gate} of "
                                f"qubit {qubit_param_pair[0]}."
                            )
                        if pulse.duration < self.min_length:
                            raise TranspilerError(
                                f"Pulse gate duration is less than {self.min_length}. "
                                "This pulse cannot be played on the specified backend. "
                                f"Please modify the duration of the custom gate pulse {pulse.name} "
                                f"which is associated with the gate {gate} of "
                                "qubit {qubit_param_pair[0]}."
                            )
