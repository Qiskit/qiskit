# This code is part of Qiskit.
#
# (C) Copyright IBM 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Set the ``layout`` property to the given layout."""
from qiskit.transpiler import Layout
from qiskit.transpiler.basepasses import AnalysisPass


class SetLayout(AnalysisPass):
    """Set the ``layout`` property to the given layout.

    This pass associates a physical qubit (int) to each virtual qubit
    of the circuit (Qubit) in increasing order.
    """

    def __init__(self, layout):
        """SetLayout initializer.

        Args:
            layout (Layout or List[int] or Dict[int, int]): the layout to set or a list to
            reorder qubits.
        """
        super().__init__()
        self.layout = layout

    def run(self, dag):
        """Run the SetLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: the original DAG.
        """
        if isinstance(self.layout, list):
            if len(self.layout) != len(dag.qubits):
                raise Exception("TODO")
            layout = Layout({phys: dag.qubits[i] for i, phys in enumerate(self.layout)})
        elif isinstance(self.layout, Layout):
            layout = self.layout.copy()
        elif self.layout is None:
            layout = None
        else:
            raise Exception("TODO")
        self.property_set["layout"] = layout
        return dag
