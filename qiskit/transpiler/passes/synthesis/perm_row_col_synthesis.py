# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Perm_row_col synthesis implementation"""

import numpy as np

from qiskit.transpiler.passes.synthesis.linear_functions_synthesis import LinearFunctionsSynthesis
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.synthesis.permrowcol import PermRowCol
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag


class PermRowColSynthesis(LinearFunctionsSynthesis):
    """Synthesize high-level objects by using permrowcol algorithm"""

    def __init__(self, coupling_map: CouplingMap):
        self._coupling_map = coupling_map

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Perform the synthesization and return the synthesized
        circuit as a dag

        Args:
            dag (DAGCircuit): dag circuit to re-synthesize

        Returns:
            DAGCircuit: re-synthesized dag circuit
        """
        for node in dag.named_nodes("cx"):
            # TODO: do something to the nodes
            pass

        # alt parity matrix of dag circuit, dtype=bool
        # parity_mat = LinearFunction(dag_to_circuit(dag)).linear

        parity_mat = np.identity(3)
        permrowcol = PermRowCol(self._coupling_map)
        circuit, perm = permrowcol.perm_row_col(parity_mat)
        return circuit_to_dag(circuit)
