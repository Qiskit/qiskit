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
import math
import os

import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
    UGate,
    PhaseGate,
    U3Gate,
    U2Gate,
    U1Gate,
    RXGate,
    RYGate,
    RZGate,
    RGate,
    SXGate,
    XGate,
)
from qiskit.circuit import Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit


logger = logging.getLogger(__name__)

NAME_MAP = {
    "u": UGate,
    "u1": U1Gate,
    "u2": U2Gate,
    "u3": U3Gate,
    "p": PhaseGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "r": RGate,
    "sx": SXGate,
    "x": XGate,
}


class Optimize1qGatesDecomposition(TransformationPass):
    """Optimize chains of single-qubit gates by combining them into a single gate.

    The decision to replace the original chain with a new resynthesis depends on:
     - whether the original chain was out of basis: replace
     - whether the original chain was in basis but resynthesis is lower error: replace
     - whether the original chain contains a pulse gate: do not replace
     - whether the original chain amounts to identity: replace with null

     Error is computed as a multiplication of the errors of individual gates on that qubit.
    """

    def __init__(self, basis=None, target=None):
        """Optimize1qGatesDecomposition initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter
                and the Euler basis. Ignored if ``target`` is also specified.
            target (Optional[Target]): The :class:`~.Target` object corresponding to the compilation
                target. When specified, any argument specified for ``basis_gates`` is ignored.
        """
        super().__init__()

        self._basis_gates = basis
        self._target = target
        self._global_decomposers = []
        self._local_decomposers_cache = {}

        if basis:
            self._global_decomposers = _possible_decomposers(set(basis))
        elif target is None:
            self._global_decomposers = _possible_decomposers(None)
            self._basis_gates = None

        self.error_map = self._build_error_map()

    def _build_error_map(self):
        if self._target is not None:
            error_map = {qubit[0]: {} for qubit in self._target.qargs if len(qubit) == 1}
            for qubit in error_map:
                gate_error = {}
                for gate in self._target:
                    if self._target[gate] is not None:
                        props = self._target[gate].get((qubit,), None)
                        if props is not None and props.error is not None:
                            gate_error[gate] = props.error
                error_map[qubit] = gate_error
            return error_map
        else:
            return None

    def _resynthesize_run(self, matrix, qubit=None, run_in_parallel=True):
        """
        Resynthesizes one 2x2 `matrix`, typically extracted via `dag.collect_1q_runs`.

        Returns the newly synthesized circuit in the indicated basis, or None
        if no synthesis routine applied.

        When multiple synthesis options are available, it prefers the one with lowest
        error when the circuit is applied to `qubit`.
        """
        if self._target:
            if qubit is not None:
                qubits_tuple = (qubit,)
            else:
                qubits_tuple = None
            if qubits_tuple in self._local_decomposers_cache:
                decomposers = self._local_decomposers_cache[qubits_tuple]
            else:
                available_1q_basis = set(self._target.operation_names_for_qargs(qubits_tuple))
                decomposers = _possible_decomposers(available_1q_basis)
                self._local_decomposers_cache[qubits_tuple] = decomposers
        else:
            decomposers = self._global_decomposers
        if self.error_map is None:
            error_map = None
        else:
            error_map = self.error_map.get(qubit, None)
        best_synth_circuit = euler_one_qubit_decomposer.unitary_to_gate_sequence(
            matrix, decomposers, error_map, run_in_parallel and len(decomposers) > 1
        )
        return best_synth_circuit

    def _gate_sequence_to_dag(self, best_synth_circuit):
        qubits = [Qubit()]
        out_dag = DAGCircuit()
        out_dag.add_qubits(qubits)
        out_dag.global_phase = best_synth_circuit.global_phase

        for gate_name, angles in best_synth_circuit:
            out_dag.apply_operation_back(NAME_MAP[gate_name](*angles), qubits)
        return out_dag

    def _substitution_checks(self, dag, old_run, new_circ, basis, qubit):
        """
        Returns `True` when it is recommended to replace `old_run` with `new_circ` over `basis`.
        """
        if new_circ is None:
            return False

        # do we even have calibrations?
        has_cals_p = dag.calibrations is not None and len(dag.calibrations) > 0
        # does this run have uncalibrated gates?
        uncalibrated_p = not has_cals_p or any(not dag.has_calibration_for(g) for g in old_run)
        # does this run have gates not in the image of ._decomposers _and_ uncalibrated?
        if basis is not None:
            uncalibrated_and_not_basis_p = any(
                g.name not in basis and (not has_cals_p or not dag.has_calibration_for(g))
                for g in old_run
            )
        else:
            # If no basis is specified then we're always in the basis
            uncalibrated_and_not_basis_p = False

        # if we're outside of the basis set, we're obligated to logically decompose.
        # if we're outside of the set of gates for which we have physical definitions,
        #    then we _try_ to decompose, using the results if we see improvement.
        return (
            uncalibrated_and_not_basis_p
            or (
                uncalibrated_p
                and _error(new_circ, self._target, qubit) < _error(old_run, self._target, qubit)
            )
            or math.isclose(_error(new_circ, self._target, qubit), 0)
        )

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        run_in_parallel = (
            os.getenv("QISKIT_IN_PARALLEL", "FALSE").upper() != "TRUE"
            or os.getenv("QISKIT_FORCE_THREADS", "FALSE").upper() == "TRUE"
        )

        runs = dag.collect_1q_runs()
        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        for run in runs:
            qubit = qubit_indices[run[0].qargs[0]]
            operator = run[0].op.to_matrix()
            for node in run[1:]:
                operator = node.op.to_matrix().dot(operator)
            new_dag = self._resynthesize_run(operator, qubit, run_in_parallel)

            if self._target is None:
                basis = self._basis_gates
            else:
                basis = self._target.operation_names_for_qargs((qubit,))

            if new_dag is not None and self._substitution_checks(dag, run, new_dag, basis, qubit):
                new_dag = self._gate_sequence_to_dag(new_dag)
                dag.substitute_node_with_dag(run[0], new_dag)
                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

        return dag


def _possible_decomposers(basis_set):
    decomposers = []
    if basis_set is None:
        decomposers = list(one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES)
    else:
        euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
        for euler_basis_name, gates in euler_basis_gates.items():
            if set(gates).issubset(basis_set):
                decomposers.append(euler_basis_name)
        if "U3" in basis_set and "U321" in basis_set:
            decomposers.remove("U3")
    return decomposers


def _error(circuit, target=None, qubit=None):
    """
    Calculate a rough error for a `circuit` that runs on a specific
    `qubit` of `target` (circuit could also be a list of DAGNodes)

    Use basis errors from target if available, otherwise use length
    of circuit as a weak proxy for error.
    """
    if target is None:
        if isinstance(circuit, (list, euler_one_qubit_decomposer.OneQubitGateSequence)):
            return len(circuit)
        else:
            return len(circuit._multi_graph) - 2
    else:
        if isinstance(circuit, euler_one_qubit_decomposer.OneQubitGateSequence):
            gate_fidelities = [
                1 - getattr(target[node[0]].get((qubit,)), "error", 0.0) for node in circuit
            ]
        elif isinstance(circuit, list):
            gate_fidelities = [
                1 - getattr(target[node.name].get((qubit,)), "error", 0.0) for node in circuit
            ]
        else:
            gate_fidelities = [
                1 - getattr(target[inst.op.name].get((qubit,)), "error", 0.0)
                for inst in circuit.op_nodes()
            ]
        gate_error = 1 - np.product(gate_fidelities)
        if gate_error == 0.0:
            if isinstance(circuit, (list, euler_one_qubit_decomposer.OneQubitGateSequence)):
                return -100 + len(circuit)
            else:
                return -100 + len(
                    circuit._multi_graph
                )  # prefer shorter circuits among those with zero error
        else:
            return gate_error
