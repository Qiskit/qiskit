# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Choose a time unit to be used in the scheduling and its following passes."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.exceptions import TranspilerError


class TimeUnitAnalysis(AnalysisPass):
    """Choose a time unit to be used in the following passes
    (e.g. scheduling pass and dynamical decoupling pass).

    """

    def __init__(self, inst_durations):
        """TrivialLayout initializer.

        Args:
            inst_durations (InstructionDurations): TO BE FILLED.
            dt (float): TO BE FILLED.

        Raises:
            TranspilerError: if invalid options
        """
        super().__init__()
        self.durations = inst_durations

    def run(self, dag):
        """Run the TimeUnitAnalysis pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Raises:
            TranspilerError: if the units are not unifiable
        """
        # TODO: implement!
        self.property_set['time_unit'] = 'dt'  # mock implementation
