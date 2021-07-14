# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesize UnitaryGates."""

from math import pi, inf
from typing import List, Union
from copy import deepcopy

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.extensions.quantum_initializer import isometry
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.circuit.library.standard_gates import iSwapGate, CXGate, CZGate, RXXGate, ECRGate
from qiskit.providers.models import BackendProperties
from qiskit.providers.exceptions import BackendPropertyError


def _choose_kak_gate(basis_gates):
    """Choose the first available 2q gate to use in the KAK decomposition."""

    kak_gate_names = {
        "cx": CXGate(),
        "cz": CZGate(),
        "iswap": iSwapGate(),
        "rxx": RXXGate(pi / 2),
        "ecr": ECRGate(),
    }

    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(kak_gate_names.keys())
    if kak_gates:
        kak_gate = kak_gate_names[kak_gates.pop()]

    return kak_gate


def _choose_euler_basis(basis_gates):
    """ "Choose the first available 1q basis to use in the Euler decomposition."""
    basis_set = set(basis_gates or [])

    for basis, gates in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            return basis

    return None


class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(
        self,
        basis_gates: List[str],
        approximation_degree: float = 1,
        coupling_map: List = None,
        backend_props: BackendProperties = None,
        pulse_optimize: Union[bool, None] = None,
        natural_direction: Union[bool, None] = None,
        synth_gates: Union[List[str], None] = None,
    ):
        """Synthesize unitaries over some basis gates.

        This pass can approximate 2-qubit unitaries given some
        approximation closeness measure (expressed as
        approximation_degree). Other unitaries are synthesized
        exactly.

        Args:
            basis_gates: List of gate names to target.
            approximation_degree: Closeness of approximation
                (0: lowest, 1: highest).
            backend_props: Properties of a backend to synthesize for
                (e.g. gate fidelities).
            pulse_optimize: Whether to optimize pulses during
                synthesis. A value of None will attempt it but fall
                back if it doesn't succeed.
            natural_direction: Whether to apply synthesis considering
                directionality of 2-qubit gates. Only applies when
                `pulse_optimize` == True. The natural direction is
                determined by first checking to see whether the
                coupling map is unidirectional.  If there is no
                coupling map or the coupling map is bidirectional,
                the gate direction with the shorter
                duration from the backend properties will be used. If
                set to True, and a natural direction can not be
                determined, raises TranspileError. If set to None, no
                exception will be raised if a natural direction can
                not be determined.
            synth_gates: List of gates to synthesize. If None and
                `pulse_optimize` is False or None, default to
                ['unitary']. If None and `pulse_optimzie` == True,
                default to ['unitary', 'swap']

        """
        super().__init__()
        self._basis_gates = basis_gates
        self._approximation_degree = approximation_degree
        self._coupling_map = coupling_map
        self._backend_props = backend_props
        self._pulse_optimize = pulse_optimize
        self._natural_direction = natural_direction
        if synth_gates:
            self._synth_gates = synth_gates
        else:
            if pulse_optimize:
                self._synth_gates = ["unitary", "swap"]
            else:
                self._synth_gates = ["unitary"]

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.

        Raises:
            TranspilerError:
                1. pulse_optimize is True but pulse optimal decomposition is not known
                   for requested basis.
                2. pulse_optimize is True and natural_direction is True but a preferred
                   gate direction can't be determined from the coupling map or the
                   relative gate lengths.
        """

        euler_basis = _choose_euler_basis(self._basis_gates)
        kak_gate = _choose_kak_gate(self._basis_gates)
        decomposer1q, decomposer2q = None, None
        if euler_basis is not None:
            decomposer1q = one_qubit_decompose.OneQubitEulerDecomposer(euler_basis)
        if kak_gate is not None:
            decomposer2q = TwoQubitBasisDecomposer(
                kak_gate, euler_basis=euler_basis, pulse_optimize=self._pulse_optimize
            )

        for node in dag.named_nodes(*self._synth_gates):
            if self._basis_gates and node.name in self._basis_gates:
                continue
            synth_dag = None
            wires = None
            if len(node.qargs) == 1:
                if decomposer1q is None:
                    continue
                synth_dag = circuit_to_dag(decomposer1q._decompose(node.op.to_matrix()))
            elif len(node.qargs) == 2:
                if decomposer2q is None:
                    continue
                synth_dag, wires = self._synth_natural_direction(node, dag, decomposer2q)
            else:
                synth_dag = circuit_to_dag(isometry.Isometry(node.op.to_matrix(), 0, 0).definition)

            dag.substitute_node_with_dag(node, synth_dag, wires=wires)

        return dag

    def _synth_natural_direction(self, node, dag, decomposer2q):
        layout = self.property_set["layout"]
        preferred_direction = None
        synth_direction = None
        physical_gate_fidelity = None
        wires = None
        dag_qubit_index = {qubit: index for index, qubit in enumerate(dag.qubits)}
        if self._natural_direction in {None, True} and layout and self._coupling_map:
            neighbors0 = self._coupling_map.neighbors(dag_qubit_index[node.qargs[0]])
            zero_one = dag_qubit_index[node.qargs[1]] in neighbors0
            neighbors1 = self._coupling_map.neighbors(dag_qubit_index[node.qargs[1]])
            one_zero = dag_qubit_index[node.qargs[0]] in neighbors1
            if zero_one and not one_zero:
                preferred_direction = [0, 1]
            if one_zero and not zero_one:
                preferred_direction = [1, 0]
        if (
            self._natural_direction in {None, True}
            and preferred_direction is None
            and layout
            and self._backend_props
        ):
            len_0_1 = inf
            len_1_0 = inf
            try:
                len_0_1 = self._backend_props.gate_length(
                    decomposer2q.gate.name,
                    [dag_qubit_index[node.qargs[0]], dag_qubit_index[node.qargs[1]]],
                )
            except BackendPropertyError:
                pass
            try:
                len_1_0 = self._backend_props.gate_length(
                    decomposer2q.gate.name,
                    [dag_qubit_index[node.qargs[1]], dag_qubit_index[node.qargs[0]]],
                )
            except BackendPropertyError:
                pass

            if len_0_1 < len_1_0:
                preferred_direction = [0, 1]
            elif len_1_0 < len_0_1:
                preferred_direction = [1, 0]
            if preferred_direction:
                physical_gate_fidelity = 1 - self._backend_props.gate_error(
                    "cx", [dag_qubit_index[node.qargs[i]] for i in preferred_direction]
                )
        if self._natural_direction is True and preferred_direction is None:
            raise TranspilerError(
                f"No preferred direction of {node.name} gate "
                "could be determined from coupling map or "
                "gate lengths."
            )
        basis_fidelity = self._approximation_degree or physical_gate_fidelity
        su4_mat = node.op.to_matrix()
        synth_circ = decomposer2q(su4_mat, basis_fidelity=basis_fidelity)
        synth_dag = circuit_to_dag(synth_circ)

        # if a natural direction exists but the synthesis is in the opposite direction,
        # resynthesize a new operator which is the original conjugated by swaps.
        # this new operator is doubly mirrored from the original and is locally equivalent.
        synth_dag_qubit_index = {qubit: index for index, qubit in enumerate(synth_dag.qubits)}
        if synth_dag.two_qubit_ops():
            synth_direction = [
                synth_dag_qubit_index[qubit] for qubit in synth_dag.two_qubit_ops()[0].qargs
            ]
        if (
            preferred_direction
            and self._pulse_optimize in {True, None}
            and synth_direction != preferred_direction
        ):
            su4_mat_mm = deepcopy(su4_mat)
            su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]]
            su4_mat_mm[:, [1, 2]] = su4_mat_mm[:, [2, 1]]
            synth_dag = circuit_to_dag(decomposer2q(su4_mat_mm, basis_fidelity=basis_fidelity))
            wires = synth_dag.wires[::-1]
        return synth_dag, wires
