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

import copy
import logging
import math

import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.converters import circuit_to_dag

logger = logging.getLogger(__name__)


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
        self.basis = None
        if basis:
            self.basis = []
            basis_set = set(basis)
            euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
            for euler_basis_name, gates in euler_basis_gates.items():
                if set(gates).issubset(basis_set):
                    basis_copy = copy.copy(self.basis)
                    for base in basis_copy:
                        # check if gates are a superset of another basis
                        # and if so, remove that basis
                        if set(euler_basis_gates[base.basis]).issubset(set(gates)):
                            self.basis.remove(base)
                        # check if the gates are a subset of another basis
                        elif set(gates).issubset(set(euler_basis_gates[base.basis])):
                            break
                    # if not a subset, add it to the list
                    else:
                        self.basis.append(one_qubit_decompose.OneQubitEulerDecomposer(
                            euler_basis_name))

    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if not self.basis:
            logger.info("Skipping pass because no basis is set")
            return dag
        runs = dag.collect_1q_runs()
        identity_matrix = np.eye(2)
        for run in runs:
            single_u3 = False
            # Don't try to optimize a single 1q gate, except for U3
            if len(run) <= 1:
                params = run[0].op.params
                # Remove single identity gates
                if len(params) > 0 and np.array_equal(run[0].op.to_matrix(),
                                                      identity_matrix):
                    dag.remove_op_node(run[0])
                    continue
                if isinstance(run[0].op, U3Gate):
                    param = float(params[0])
                    if math.isclose(param, 0, rel_tol=0, abs_tol=1e-12) or math.isclose(
                            param, np.pi/2, abs_tol=1e-12, rel_tol=0):
                        single_u3 = True
                    else:
                        continue
                else:
                    continue

            new_circs = []
            operator = run[0].op.to_matrix()
            for gate in run[1:]:
                operator = gate.op.to_matrix().dot(operator)
            for decomposer in self.basis:
                new_circs.append(decomposer._decompose(operator))
            if new_circs:
                new_circ = min(new_circs, key=len)
                if (len(run) > len(new_circ) or (single_u3 and
                                                 new_circ.data[0][0].name != 'u3')):
                    new_dag = circuit_to_dag(new_circ)
                    dag.substitute_node_with_dag(run[0], new_dag)
                    # Delete the other nodes in the run
                    for current_node in run[1:]:
                        dag.remove_op_node(current_node)
        return dag
