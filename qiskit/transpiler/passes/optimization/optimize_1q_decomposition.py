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

import numpy as np

from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.converters import circuit_to_dag

logger = logging.getLogger(__name__)


class Optimize1qGatesDecomposition(TransformationPass):
    """Optimize chains of single-qubit gates by combining them into a single gate."""

    def __init__(self, basis=None, target=None):
        """Optimize1qGatesDecomposition initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter
                and the Euler basis.
            target (Target): The target representing the backend. If specified
                this will be used instead of the ``basis_gates`` parameter

        """
        super().__init__()

        # change name so as to not confuse with target basis
        self._basis_set = basis
        self._decomposers = None
        self._target = target

        # try to get the op names of the target to see what are
        # the needed decomposers
        if basis or target:
            self._decomposers = {}
            euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES

            if basis:
                basis_set = set(basis)
            # get all the possible operations that could be
            # performed in the target

            # target basis supersedes original
            if target:
                basis_set = target.operation_names

            for euler_basis_name, gates in euler_basis_gates.items():
                if set(gates).issubset(basis_set):
                    basis_copy = copy.copy(self._decomposers)
                    for base in basis_copy.keys():
                        # check if gates are a superset of another basis
                        if set(base).issubset(set(gates)):
                            # if so, remove that basis
                            del self._decomposers[base]
                        # check if the gates are a subset of another basis
                        elif set(gates).issubset(set(base)):
                            # if so, don't bother
                            break
                    # if not a subset, add it to the list
                    else:
                        self._decomposers[
                            tuple(gates)
                        ] = one_qubit_decompose.OneQubitEulerDecomposer(euler_basis_name)

    def _resynthesize_run(self, run):
        """
        Resynthesizes one `run`, typically extracted via `dag.collect_1q_runs`.

        Returns (basis, circuit) containing the newly synthesized circuit in the indicated basis, or
        (None, None) if no synthesis routine applied.
        """

        operator = run[0].op.to_matrix()
        for gate in run[1:]:
            operator = gate.op.to_matrix().dot(operator)

        new_circs = {k: v._decompose(operator) for k, v in self._decomposers.items()}
        new_basis, new_circ = None, None
        if len(new_circs) > 0:
            new_basis, new_circ = min(new_circs.items(), key=lambda x: len(x[1]))

        return new_basis, new_circ

    def _substitution_checks(self, dag, old_run, new_circ, new_basis, qubit_map):
        """
        Returns `True` when it is recommended to replace `old_run` with `new_circ`.
        """

        if new_circ is None:
            return False

        # do we even have calibrations?
        has_cals_p = dag.calibrations is not None and len(dag.calibrations) > 0
        # is this run in the target set of this particular decomposer and also uncalibrated?
        rewriteable_and_in_basis_p = all(
            g.name in new_basis and (not has_cals_p or not dag.has_calibration_for(g))
            for g in old_run
        )
        # does this run have uncalibrated gates?
        uncalibrated_p = not has_cals_p or any(not dag.has_calibration_for(g) for g in old_run)
        # does this run have gates not in the image of ._decomposers _and_ uncalibrated?
        def _not_in_basis(gate):
            """To check if gate in basis or not

            Args:
                gate (Gate): gate to check

            Returns:
                Bool : whether Gate is supported by target or in basis set
            """
            if self._target:
                in_target = self._target.instruction_supported(
                    gate.name, tuple(qubit_map[bit] for bit in gate.qargs)
                )
                return not in_target
            if self._basis_set:
                in_basis = gate.name in self._basis_set
                return not in_basis

            return True  # basis is None and target is None too

        uncalibrated_and_not_basis_p = any(
            _not_in_basis(g) and (not has_cals_p or not dag.has_calibration_for(g)) for g in old_run
        )

        if rewriteable_and_in_basis_p and len(old_run) < len(new_circ):
            # NOTE: This is short-circuited on calibrated gates, which we're timid about
            #       reducing.
            logger.debug(
                "Resynthesized \n\n"
                + "\n".join([str(node.op) for node in old_run])
                + "\n\nand got\n\n"
                + "\n".join([str(node[0]) for node in new_circ])
                + f"\n\nbut the original was native (for {self._basis_set}) and the new value "
                "is longer.  This indicates an efficiency bug in synthesis.  Please report it by "
                "opening an issue here: "
                "https://github.com/Qiskit/qiskit-terra/issues/new/choose",
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
        return (
            uncalibrated_and_not_basis_p
            or (uncalibrated_p and len(old_run) > len(new_circ))
            or isinstance(old_run[0].op, U3Gate)
        )

    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if self._decomposers is None:
            logger.info("Skipping pass because no basis is set")
            return dag

        runs = dag.collect_1q_runs()

        # required for checking instruction support
        qubit_map = None
        if self._target:
            qubit_map = {qubit: index for index, qubit in enumerate(dag.qubits)}

        for run in runs:
            # SPECIAL CASE: Don't bother to optimize single U3 gates which are in the basis set.
            #     The U3 decomposer is only going to emit a sequence of length 1 anyhow.
            if isinstance(run[0].op, U3Gate) and len(run) == 1:
                # try to optimize
                if self._target:
                    if self._target.instruction_supported(
                        "u3", tuple(qubit_map[bit] for bit in run[0].qargs)
                    ):
                        if np.allclose(run[0].op.to_matrix(), np.eye(2), 1e-15, 0):
                            dag.remove_op_node(run[0])
                            continue
                        # if u2 or u1 supported on this qubit, we may try decomposition
                        u21_in_target = "u2" in self._target.instruction_supported(
                            "u2", tuple(qubit_map[bit] for bit in run[0].qargs)
                        ) or "u1" in self._target.instruction_supported(
                            "u1", tuple(qubit_map[bit] for bit in run[0].qargs)
                        )
                        if not u21_in_target:
                            continue

                elif "u3" in self._basis_set:
                    # Toss U3 gates equivalent to the identity; there we get off easy.
                    if np.allclose(run[0].op.to_matrix(), np.eye(2), 1e-15, 0):
                        dag.remove_op_node(run[0])
                        continue
                    # if u21 supported, try to decompose, else continue
                    u21_in_basis = "u2" in self._basis_set or "u1" in self._basis_set

                    if not u21_in_basis:
                        continue

            new_basis, new_circ = self._resynthesize_run(run)

            if new_circ is not None and self._substitution_checks(
                dag, run, new_circ, new_basis, qubit_map
            ):
                new_dag = circuit_to_dag(new_circ)
                dag.substitute_node_with_dag(run[0], new_dag)
                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

        return dag
