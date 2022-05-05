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

"""Check if a property reached a fixed point."""

from qiskit.transpiler.basepasses import AnalysisPass


class ContainsInstruction(AnalysisPass):
    """An analysis pass to detect if the DAG contains a specific instruction.

    This pass takes in a single instruction name for example ``'delay'`` and
    will set the property set ``contains_delay`` to ``True`` if the DAG contains
    that instruction and ``False`` if it does not.
    """

    def __init__(self, instruction_name):
        """ContainsInstruction initializer.

        Args:
            instruction_name (str): The instruction to check if it's in the
                DAG. The output in the property set is set to ``contains_``
                prefixed on the value for this parameter.
        """
        super().__init__()
        self._instruction_name = instruction_name

    def run(self, dag):
        """Run the ContainsInstruction pass on dag."""
        self.property_set[f"contains_{self._instruction_name}"] = (
            self._instruction_name in dag._op_names
        )
