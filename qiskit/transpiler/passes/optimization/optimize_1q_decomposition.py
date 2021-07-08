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
import warnings

import numpy as np

from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import one_qubit_decompose
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
        self._target_basis = basis
        self._decomposers = None
        if basis:
            self._decomposers = []
            basis_set = set(basis)
            euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
            for euler_basis_name, gates in euler_basis_gates.items():
                if set(gates).issubset(basis_set):
                    basis_copy = copy.copy(self._decomposers)
                    for base in basis_copy:
                        # check if gates are a superset of another basis
                        # and if so, remove that basis
                        if set(euler_basis_gates[base.basis]).issubset(set(gates)):
                            self._decomposers.remove(base)
                        # check if the gates are a subset of another basis
                        elif set(gates).issubset(set(euler_basis_gates[base.basis])):
                            break
                    # if not a subset, add it to the list
                    else:
                        self._decomposers.append(
                            one_qubit_decompose.OneQubitEulerDecomposer(euler_basis_name)
                        )

    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if not self._decomposers:
            logger.info("Skipping pass because no basis is set")
            return dag
        runs = dag.collect_1q_runs()
        for run in runs:
            # SPECIAL CASE: Don't bother to optimize single U3 gates which are in the basis set.
            #     The U3 decomposer is only going to emit a sequence of length 1 anyhow.
            if "u3" in self._target_basis and len(run) == 1 and isinstance(run[0].op, U3Gate):
                # Toss U3 gates equivalent to the identity; there we get off easy.
                if np.array_equal(run[0].op.to_matrix(), np.eye(2)):
                    dag.remove_op_node(run[0])
                    continue
                # We might rewrite into lower `u`s if they're available.
                if "u2" not in self._target_basis and "u1" not in self._target_basis:
                    continue

            new_circs = []
            operator = run[0].op.to_matrix()
            for gate in run[1:]:
                operator = gate.op.to_matrix().dot(operator)
            for decomposer in self._decomposers:
                new_circs.append(decomposer._decompose(operator))
            if new_circs:
                new_circ = min(new_circs, key=len)

                # do we even have calibrations?
                has_cals_p = dag.calibrations is not None and len(dag.calibrations) > 0
                # is this run all in the target set and also uncalibrated?
                rewriteable_and_in_basis_p = all(
                    g.name in self._target_basis
                    and (not has_cals_p or not dag.has_calibration_for(g))
                    for g in run
                )
                # does this run have uncalibrated gates?
                uncalibrated_p = not has_cals_p or any(not dag.has_calibration_for(g) for g in run)
                # does this run have gates not in the image of ._decomposers _and_ uncalibrated?
                uncalibrated_and_not_basis_p = any(
                    g.name not in self._target_basis
                    and (not has_cals_p or not dag.has_calibration_for(g))
                    for g in run
                )

                if rewriteable_and_in_basis_p and len(run) < len(new_circ):
                    # NOTE: This is short-circuited on calibrated gates, which we're timid about
                    #       reducing.
                    warnings.warn(
                        f"Resynthesized {run} and got {new_circ}, "
                        f"but the original was native and the new value is longer.  This "
                        f"indicates an efficiency bug in synthesis.  Please report it by "
                        f"opening an issue here: "
                        f"https://github.com/Qiskit/qiskit-terra/issues/new/choose",
                        stacklevel=2,
                    )
                # if we're outside of the basis set, we're obligated to logically decompose.
                # if we're outside of the set of gates for which we have physical definitions,
                #    then we _try_ to decompose, using the results if we see improvement.
                # NOTE: Here we use circuit length as a weak proxy for "improvement"; in reality,
                #       we care about something more like fidelity at runtime, which would mean,
                #       e.g., a preference for `RZGate`s over `RXGate`s.  In fact, users sometimes
                #       express a preference for a "canonical form" of a circuit, which may come in
                #       the form of some parameter values, also not visible at the level of circuit
                #       length.  Since we don't have a framework for the caller to programmatically
                #       express what they want here, we include some special casing for particular
                #       gates which we've promised to normalize --- but this is fragile and should
                #       ultimately be done away with.
                if (
                    uncalibrated_and_not_basis_p
                    or (uncalibrated_p and len(run) > len(new_circ))
                    or isinstance(run[0].op, U3Gate)
                ):
                    new_dag = circuit_to_dag(new_circ)
                    dag.substitute_node_with_dag(run[0], new_dag)
                    # Delete the other nodes in the run
                    for current_node in run[1:]:
                        dag.remove_op_node(current_node)
        return dag
