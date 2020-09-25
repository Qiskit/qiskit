# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Choose a time unit to be used in the scheduling and its following passes."""
from typing import Set

from qiskit.circuit import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations


class TimeUnitAnalysis(AnalysisPass):
    """Choose a time unit to be used in the following passes
    (e.g. scheduling pass and dynamical decoupling pass).

    If dt (dt in seconds) is known to transpiler, the unit 'dt' is chosen. Otherwise,
    the unit to be selected depends on what units are used in delays and instruction durations:
    * 's': if they are all in SI units.
    * 'dt': if they are all in the unit 'dt'.
    * raise error: if they are a mix of SI units and 'dt'.
    """

    def __init__(self, inst_durations: InstructionDurations):
        """TimeUnitAnalysis initializer.

        Args:
            inst_durations (InstructionDurations): A dictionary of durations of instructions.
        """
        super().__init__()
        self.inst_durations = inst_durations

    def run(self, dag: DAGCircuit):
        """Run the TimeUnitAnalysis pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Raises:
            TranspilerError: if the units are not unifiable
        """
        if self.inst_durations.dt is not None:
            self.property_set['time_unit'] = 'dt'
        else:
            # Check what units are used in delays and other instructions: dt or SI or mixed
            units_delay = self._units_used_in_delays(dag)
            if self._unified(units_delay) == "mixed":
                raise TranspilerError("Fail to unify time units in delays. SI units "
                                      "and dt unit must not be mixed when dt is not supplied.")
            units_other = self.inst_durations.units_used()
            if self._unified(units_other) == "mixed":
                raise TranspilerError("Fail to unify time units in instruction_durations. SI units "
                                      "and dt unit must not be mixed when dt is not supplied.")

            unified_unit = self._unified(units_delay | units_other)
            if unified_unit == "SI":
                self.property_set['time_unit'] = 's'
            elif unified_unit == "dt":
                self.property_set['time_unit'] = 'dt'
            else:
                raise TranspilerError("Fail to unify time units. SI units "
                                      "and dt unit must not be mixed when dt is not supplied.")

    @staticmethod
    def _units_used_in_delays(dag: DAGCircuit) -> Set[str]:
        units_used = set()
        for node in dag.op_nodes(op=Delay):
            units_used.add(node.op.unit)
        return units_used

    @staticmethod
    def _unified(unit_set: Set[str]) -> str:
        if not unit_set:
            return "dt"

        if len(unit_set) == 1 and 'dt' in unit_set:
            return "dt"

        all_si = True
        for unit in unit_set:
            if not unit.endswith('s'):
                all_si = False
                break

        if all_si:
            return "SI"

        return "mixed"
