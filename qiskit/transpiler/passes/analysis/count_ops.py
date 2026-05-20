# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Count the operations in a DAG circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import AnalysisPass


class CountOps(AnalysisPass):
    """Count the operations in a DAG circuit.

    The result is saved in ``property_set['count_ops']`` as an integer.
    """

    def __init__(self, *, recurse: bool = True) -> None:
        """
        Args:
            recurse: If ``True`` (default), recursively count operations
                inside control-flow blocks.
        """
        super().__init__()
        self.recurse = recurse

    def run(self, dag: DAGCircuit) -> None:
        """Run the CountOps pass on ``dag``."""
        self.property_set["count_ops"] = dag.count_ops(recurse=self.recurse)


class GateCount(AnalysisPass):
    """Count the number of specified gates and store it in the specified key in the property set.

    Stores the sum of all specified gates in ``property_set[key]``. If a gate
    string is invalid, the gate is ignored and no error is raised.

    For example::

        t_count = GateCount(gates=["t", "tdg"], key="t_count")
        rz_count = GateCount(gates=["rz"], key="rz_count")
    """

    def __init__(self, *, gates, key, recurse: bool = True) -> None:
        """
        Args:
            gates: The gates to count.
            key: The key to store it in.
            recurse: If ``True`` (default), recursively count operations
                inside control-flow blocks.  Note that this does not multiply by
                the number of times a control-flow block is executed.
        """
        super().__init__()
        self.gates = gates
        self.key = key
        self.recurse = recurse

    def run(self, dag: DAGCircuit) -> None:
        ops = dag.count_ops(recurse=self.recurse)
        count = sum(ops.get(gate, 0) for gate in self.gates)
        self.property_set[self.key] = count
