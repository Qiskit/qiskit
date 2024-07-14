# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Collect sequences of uninterrupted gates acting on 1 qubit."""

from qiskit.transpiler.basepasses import AnalysisPass


class Collect1qRuns(AnalysisPass):
    """Collect one-qubit subcircuits."""

    def run(self, dag):
        """Run the Collect1qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['run_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set["run_list"] = dag.collect_1q_runs()
        return dag
