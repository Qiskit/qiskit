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

"""Add user provided calibrations."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap, CalibrationPublisher
from qiskit.transpiler.basepasses import TransformationPass


class PulseGates(TransformationPass):
    """Pulse gate adding pass.

    This pass adds gate calibrations to a quantum circuit.
    In QASM3 [1], the gate calibration can be encapsulated as ``defcal`` entry that declares
    the hardware implementation of ``gate`` in specified grammar, such as OpenPulse.

    This pass checks each DAG circuit node and acquires a corresponding schedule from
    the instruction schedule map object provided by the target backend.
    This mapping object returns a schedule with "publisher" metadata which is an integer Enum
    value representing who created the gate schedule.
    If the gate schedule is provided by a client, this pass attaches this to
    the DAG circuit as a ``defcal`` entry.

    This pass allows users to easily override quantum circuit with custom gate definition
    without directly dealing with those schedules.

    References
        * [1] OpenQASM 3: A broader and deeper quantum assembly language
          https://arxiv.org/abs/2104.14722
    """

    def __init__(
            self,
            inst_map: InstructionScheduleMap,
    ):
        """Create new pass.

        Args:
            inst_map: Instruction schedule map that user may override.
        """
        super().__init__()
        self.inst_map = inst_map

    def run(self, dag: DAGCircuit):
        """Run the TimeUnitAnalysis pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to attach calibrations.

        Returns:
            DAGCircuit: DAG with calibration definitions.
        """
        already_defined = set()
        for node in dag.gate_nodes():
            qubits = tuple(dag.qubits.index(q) for q in node.qargs)
            key = [node.op.name, qubits, *node.op.params]
            # check if calibration is already defined, or already defined
            if not dag.has_calibration_for(node) and key not in already_defined:
                sched = self.inst_map.get(*key)
                publisher = sched.metadata.get("publisher", CalibrationPublisher.Qiskit)
                if publisher != CalibrationPublisher.BackendProvider:
                    # likely user provided calibration, add schedule
                    dag.add_calibration(gate=node.op, qubits=qubits, schedule=sched)
                else:
                    # to avoid next check, this calibration is backend default
                    already_defined.add(key)
        return dag
