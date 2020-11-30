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

from itertools import groupby
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
            basis_set = set(basis)
            for basis_name, gates in self.euler_basis_names.items():
                if set(gates).issubset(basis_set):
                    self.basis = basis_name
                    break

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
        decomposer = OneQubitEulerDecomposer(self.basis)
        runs = dag.collect_runs(self.euler_basis_names[self.basis])
        runs = _split_runs_on_parameters(runs)
        for run in runs:
            if len(run) <= 1:
                params = run[0].op.params
                # Remove single identity gates
                if run[0].op.name in self.euler_basis_names[self.basis] and len(
                        params) > 0 and np.array_equal(run[0].op.to_matrix(),
                                                       np.eye(2)):
                    dag.remove_op_node(run[0])
                # Don't try to optimize a single 1q gate
                continue
            q = QuantumRegister(1, "q")
            qc = QuantumCircuit(1)
            for gate in run:
                qc.append(gate.op, [q[0]], [])

            operator = Operator(qc)
            new_circ = decomposer(operator)
            new_dag = circuit_to_dag(new_circ)
            dag.substitute_node_with_dag(run[0], new_dag)
            # Delete the other nodes in the run
            for current_node in run[1:]:
                dag.remove_op_node(current_node)
        return dag


def _split_runs_on_parameters(runs):
    """Finds runs containing parameterized gates and splits them into sequential
    runs excluding the parameterized gates.
    """

    out = []
    for run in runs:
        groups = groupby(run, lambda x: x.op.is_parameterized())

        for group_is_parameterized, gates in groups:
            if not group_is_parameterized:
                out.append(list(gates))

    return out
