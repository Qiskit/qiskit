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

# When expanding the list of supported gates this needs to updated in
# lockstep with the VALID_BASES constant in src/euler_one_qubit_decomposer.rs
# and the global variables in qiskit/quantum_info/synthesis/one_qubit_decompose.py
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
            error_map = euler_one_qubit_decomposer.OneQubitGateErrorMap(self._target.num_qubits)
            for qubit in range(self._target.num_qubits):
                gate_error = {}
                for gate, gate_props in self._target.items():
                    if gate_props is not None:
                        props = gate_props.get((qubit,), None)
                        if props is not None and props.error is not None:
                            gate_error[gate] = props.error
                error_map.add_qubit(gate_error)
            return error_map
        else:
            return None

    def _resynthesize_run(self, matrix, qubit=None):
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
        best_synth_circuit = euler_one_qubit_decomposer.unitary_to_gate_sequence(
            matrix,
            decomposers,
            qubit,
            self.error_map,
        )
        return best_synth_circuit

    def _gate_sequence_to_dag(self, best_synth_circuit):
        qubits = (Qubit(),)
        out_dag = DAGCircuit()
        out_dag.add_qubits(qubits)
        out_dag.global_phase = best_synth_circuit.global_phase

        for gate_name, angles in best_synth_circuit:
            out_dag.apply_operation_back(NAME_MAP[gate_name](*angles), qubits, check=False)
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
            or (uncalibrated_p and self._error(new_circ, qubit) < self._error(old_run, qubit))
            or math.isclose(self._error(new_circ, qubit)[0], 0)
        )

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        runs = dag.collect_1q_runs()
        for run in runs:
            qubit = dag.find_bit(run[0].qargs[0]).index
            operator = run[0].op.to_matrix()
            for node in run[1:]:
                operator = node.op.to_matrix().dot(operator)
            best_circuit_sequence = self._resynthesize_run(operator, qubit)

            if self._target is None:
                basis = self._basis_gates
            else:
                basis = self._target.operation_names_for_qargs((qubit,))

            if best_circuit_sequence is not None and self._substitution_checks(
                dag, run, best_circuit_sequence, basis, qubit
            ):
                new_dag = self._gate_sequence_to_dag(best_circuit_sequence)
                dag.substitute_node_with_dag(run[0], new_dag)
                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

        return dag

    def _error(self, circuit, qubit):
        """
        Calculate a rough error for a `circuit` that runs on a specific
        `qubit` of `target` (`circuit` can either be an OneQubitGateSequence
        from Rust or a list of DAGOPNodes).

        Use basis errors from target if available, otherwise use length
        of circuit as a weak proxy for error.
        """
        if isinstance(circuit, euler_one_qubit_decomposer.OneQubitGateSequence):
            return euler_one_qubit_decomposer.compute_error_one_qubit_sequence(
                circuit, qubit, self.error_map
            )
        else:
            circuit_list = [(x.op.name, []) for x in circuit]
            return euler_one_qubit_decomposer.compute_error_list(
                circuit_list, qubit, self.error_map
            )


def _possible_decomposers(basis_set):
    decomposers = []
    if basis_set is None:
        decomposers = list(one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES)
    else:
        euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
        for euler_basis_name, gates in euler_basis_gates.items():
            if set(gates).issubset(basis_set):
                decomposers.append(euler_basis_name)
        # If both U3 and U321 are in decomposer list only run U321 because
        # in worst case it will produce the same U3 output, but in the general
        # case it will use U2 and U1 which will be more efficient.
        if "U3" in decomposers and "U321" in decomposers:
            decomposers.remove("U3")
        # If both ZSX and ZSXX are in decomposer list only run ZSXX because
        # in the worst case it will produce the same output, but in the general
        # case it will simplify X rotations to use X gate instead of multiple
        # SX gates and be more efficient. Running multiple decomposers in this
        # case will just waste time.
        if "ZSX" in decomposers and "ZSXX" in decomposers:
            decomposers.remove("ZSX")
    return decomposers
