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

from collections import defaultdict
import copy
import logging

import numpy as np

from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers.models import BackendProperties
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.synthesis import one_qubit_decompose

logger = logging.getLogger(__name__)

DEFAULT_ATOL = 1.0e-12
"""Threshold for fidelity drop in re-synthesized sequences before emitting a warning."""


class Optimize1qGatesDecomposition(TransformationPass):
    """Optimize chains of single-qubit gates by combining them into a single gate."""

    def __init__(
        self,
        basis=None,
        backend_properties: BackendProperties = None,
    ):
        """Optimize1qGatesDecomposition initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects of this
                pass, the basis is the set intersection between the `basis` parameter and the Euler
                basis.
        """
        super().__init__()

        self._target_basis = basis
        self._decomposers = None
        self._backend_properties = backend_properties
        self._memoized_fidelities_dicts = {}

        if basis:
            self._decomposers = {}
            basis_set = set(basis)
            euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
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
                        decomposer = one_qubit_decompose.OneQubitEulerDecomposer(euler_basis_name)
                        self._decomposers[tuple(gates)] = decomposer

    def _generate_fidelities_dict(self, qubit, default=1.0):
        """
        Produces a dictionary mapping gate names to reported fidelities for the indicated `qubit`.
        Reports `default` when the gate name has no measurement present.
        """

        memoized_value = self._memoized_fidelities_dicts.get(qubit)
        if memoized_value is not None:
            return memoized_value

        # invert the key order in _backend_properties's fidelities dictionary
        if self._backend_properties is None:
            memoized_value = defaultdict(lambda: default)
        else:
            memoized_value = defaultdict(
                lambda: default,
                **{
                    x["gate"]: (
                        1 - next(y["value"] for y in x["parameters"] if y["name"] == "gate_error")
                    )
                    for x in self._backend_properties.to_dict()["gates"]
                    if x["qubits"] == [qubit]
                },
            )

        self._memoized_fidelities_dicts[qubit] = memoized_value
        return memoized_value

    def _resynthesize_run(self, dag, run, atol=DEFAULT_ATOL):
        """
        Resynthesizes one `run`, typically extracted via `dag.collect_1q_runs`.

        Returns (basis, circuit) containing the newly synthesized circuit in the indicated basis, or
        (None, None) if no synthesis routine applied.
        """

        operator = run[0].op.to_matrix()
        for gate in run[1:]:
            operator = gate.op.to_matrix().dot(operator)
        operatorH = operator.transpose(1, 0).conjugate()

        qubit = next(
            (index for (index, qubit) in enumerate(dag.qubits) if qubit == run[0].qargs[0]), None
        )
        fidelities = self._generate_fidelities_dict(qubit)

        def implementation_infidelity(circuit):
            circuit_operator = np.eye(2, dtype=complex)
            for gate, _, _ in circuit:
                circuit_operator = gate.to_matrix().dot(circuit_operator)

            approx_cost = (4 - np.abs(np.trace(circuit_operator @ operatorH)) ** 2) / 6
            op_cost = sum(1 - fidelities[x[0].name] for x in circuit)
            return op_cost + approx_cost

        candidate_circs = {
            k: v._decompose(operator, fidelity_mapping=fidelities)
            for k, v in self._decomposers.items()
        }

        chosen_basis, chosen_circ, chosen_infidelity = None, None, 1.0
        for candidate_basis, candidate_circ in candidate_circs.items():
            candidate_infidelity = implementation_infidelity(candidate_circ)
            if candidate_infidelity < chosen_infidelity - atol:
                chosen_basis, chosen_circ, chosen_infidelity = (
                    candidate_basis,
                    candidate_circ,
                    candidate_infidelity,
                )

        return chosen_basis, chosen_circ

    def _substitution_checks(self, dag, old_run, new_circ, new_basis, atol=DEFAULT_ATOL):
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
        uncalibrated_and_not_basis_p = any(
            g.name not in self._target_basis and (not has_cals_p or not dag.has_calibration_for(g))
            for g in old_run
        )

        old_infidelity = 0
        new_infidelity = 0
        # look up physical qubit index in layout
        # (this snippet is adapted from UnitarySynthesis._synth_natural_direction)
        qubit = next(
            (index for (index, qubit) in enumerate(dag.qubits) if qubit == old_run[0].qargs[0]),
            None,
        )

        if self._backend_properties is not None:
            try:
                for op in old_run:
                    if not isinstance(op, DAGOpNode):
                        continue
                    old_infidelity += self._backend_properties.gate_error(op.op.name, [qubit])
            except BackendPropertyError:
                old_infidelity += float("inf")

            for instr, _, _ in new_circ.data:
                try:
                    new_infidelity += self._backend_properties.gate_error(instr.name, [qubit])
                except BackendPropertyError:
                    pass

        # incorporate a possible trace distance from approximate synthesis
        operator = old_run[0].op.to_matrix()
        for gate in old_run[1:]:
            operator = gate.op.to_matrix().dot(operator)
        decomp_unitary = new_circ[0][0].to_matrix() if new_circ else np.eye(2)
        for gate, _, _ in new_circ[1:]:
            decomp_unitary = gate.to_matrix().dot(decomp_unitary)
        trace_pairing = np.trace(decomp_unitary @ np.conj(operator.data).transpose(1, 0))
        new_infidelity += (4 - abs(trace_pairing) ** 2) / 6

        if rewriteable_and_in_basis_p and (
            (old_infidelity + atol < new_infidelity)
            or (abs(old_infidelity - new_infidelity) < atol and len(old_run) < len(new_circ))
        ):
            # NOTE: This is short-circuited on calibrated gates, which we're timid about reducing.
            warnings.warn(
                "Resynthesized \n\n"
                + "\n".join([str(node.op) for node in old_run])
                + "\n\nand got\n\n"
                + "\n".join([str(node[0]) for node in new_circ])
                + f"\n\nbut the original was native (for {self._target_basis}) and the new value "
                f"has worse estimated infidelity (old {old_infidelity} vs new {new_infidelity}).  "
                f"This indicates an efficiency bug in synthesis.  Please report it by opening an issue here: "
                f"https://github.com/Qiskit/qiskit-terra/issues/new/choose",
            )

        # if we're outside of the basis set, we're obligated to logically decompose.
        # if we're outside of the set of gates for which we have physical definitions, then we _try_
        #    to decompose, using the results if we see improvement.
        return (
            uncalibrated_and_not_basis_p
            or (
                uncalibrated_p
                and (
                    (new_infidelity < old_infidelity - atol)
                    or (
                        abs(new_infidelity - old_infidelity) < atol and len(new_circ) < len(old_run)
                    )
                )
            )
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
        for run in runs:
            # SPECIAL CASE: Don't bother to optimize single U3 gates which are in the basis set.
            #     The U3 decomposer is only going to emit a sequence of length 1 anyhow.
            if "u3" in self._target_basis and len(run) == 1 and isinstance(run[0].op, U3Gate):
                # Toss U3 gates equivalent to the identity; there we get off easy.
                if np.allclose(run[0].op.to_matrix(), np.eye(2), 1e-15, 0):
                    dag.remove_op_node(run[0])
                    continue
                # We might rewrite into lower `u`s if they're available.
                if "u2" not in self._target_basis and "u1" not in self._target_basis:
                    continue

            new_basis, new_circ = self._resynthesize_run(dag, run)

            if new_circ is not None and self._substitution_checks(dag, run, new_circ, new_basis):
                new_dag = circuit_to_dag(new_circ)
                dag.substitute_node_with_dag(run[0], new_dag)
                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

        return dag
