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

"""Optimize chains of single-qubit gates using Euler 1q decomposer"""

import logging

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.converters import circuit_to_dag

LOG = logging.getLogger(__name__)


class Optimize1qGatesDecomposition(TransformationPass):
    """Optimize chains of single-qubit gates by combining them into a single gate."""

    def __init__(self, basis=None):
        """Optimize1qGatesDecomposition initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter
                and the Euler basis.
        """
        super().__init__()
        self.euler_basis_names = {
            'U3': ['u3'],
            'U': ['u'],
            'PSX': ['p', 'sx'],
            'U1X': ['u1', 'rx'],
            'RR': ['r'],
            'ZYZ': ['rz', 'ry'],
            'ZXZ': ['rz', 'rx'],
            'XYX': ['rx', 'ry'],
            'ZSX': ['rz', 'sx'],
        }
        self.basis = None
        if basis:
            self.basis = []
            basis_set = set(basis)
            for basis_name, gates in self.euler_basis_names.items():
                if set(gates).issubset(basis_set):
                    self.basis.append(OneQubitEulerDecomposer(basis_name))

    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if not self.basis:
            LOG.info("Skipping pass because no basis is set")
            return dag
        runs = dag._collect_1q_runs()
        for run in runs:
            # Don't try to optimize a single 1q gate
            if len(run) <= 1:
                params = run[0].op.params
                # Remove single identity gates
                if len(params) > 0 and np.array_equal(run[0].op.to_matrix(),
                                                      np.eye(2)):
                    dag.remove_op_node(run[0])
                continue

            new_circs = []
            q = QuantumRegister(1, "q")
            qc = QuantumCircuit(1)
            for gate in run:
                qc._append(gate.op, [q[0]], [])
            operator = Operator(qc)
            for decomposer in self.basis:
                new_circs.append(decomposer(operator))
            if new_circs:
                new_circ = min(new_circs, key=lambda circ: circ.depth())
                if qc.depth() > new_circ.depth():
                    new_dag = circuit_to_dag(new_circ)
                    dag.substitute_node_with_dag(run[0], new_dag)
                    # Delete the other nodes in the run
                    for current_node in run[1:]:
                        dag.remove_op_node(current_node)
        return dag
