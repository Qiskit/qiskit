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
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.extensions.quantum_initializer import isometry
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.circuit.library.standard_gates import iSwapGate, CXGate, CZGate, RXXGate, ECRGate
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.providers.models import BackendProperties


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


def _choose_bases(basis_gates, basis_dict=None):
    """Find the matching basis string keys from the list of basis gates from the backend."""
    if basis_gates is None:
        basis_set = set()
    else:
        basis_set = set(basis_gates)

    if basis_dict is None:
        basis_dict = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES

    out_basis = []
    for basis, gates in basis_dict.items():
        if set(gates).issubset(basis_set):
            out_basis.append(basis)

    return out_basis


class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(
        self,
        basis_gates: List[str],
        approximation_degree: float = 1,
        coupling_map: CouplingMap = None,
        backend_props: BackendProperties = None,
        pulse_optimize: Union[bool, None] = None,
        natural_direction: Union[bool, None] = None,
        synth_gates: Union[List[str], None] = None,
        method: str = "default",
        min_qubits: int = None,
    ):
        """Synthesize unitaries over some basis gates.

        This pass can approximate 2-qubit unitaries given some
        approximation closeness measure (expressed as
        approximation_degree). Other unitaries are synthesized
        exactly.

        Args:
            basis_gates (list[str]): List of gate names to target.
            approximation_degree (float): Closeness of approximation
                (0: lowest, 1: highest).
            coupling_map (CouplingMap): the coupling map of the backend
                in case synthesis is done on a physical circuit. The
                directionality of the coupling_map will be taken into
                account if pulse_optimize is True/None and natural_direction
                is True/None.
            backend_props (BackendProperties): Properties of a backend to
                synthesize for (e.g. gate fidelities).
            pulse_optimize (bool): Whether to optimize pulses during
                synthesis. A value of None will attempt it but fall
                back if it doesn't succeed. A value of True will raise
                an error if pulse-optimized synthesis does not succeed.
            natural_direction (bool): Whether to apply synthesis considering
                directionality of 2-qubit gates. Only applies when
                `pulse_optimize` is True or None. The natural direction is
                determined by first checking to see whether the
                coupling map is unidirectional.  If there is no
                coupling map or the coupling map is bidirectional,
                the gate direction with the shorter
                duration from the backend properties will be used. If
                set to True, and a natural direction can not be
                determined, raises TranspileError. If set to None, no
                exception will be raised if a natural direction can
                not be determined.
            synth_gates (list[str]): List of gates to synthesize. If None and
                `pulse_optimize` is False or None, default to
                ['unitary']. If None and `pulse_optimzie` == True,
                default to ['unitary', 'swap']
            method (str): The unitary synthesis method plugin to use.
            min_qubits: The minimum number of qubits in the unitary to synthesize. If this is set
                and the unitary is less than the specified number of qubits it will not be
                synthesized.
        """
        super().__init__()
        self._basis_gates = set(basis_gates or ())
        self._approximation_degree = approximation_degree
        self._min_qubits = min_qubits
        self.method = method
        self.plugins = plugin.UnitarySynthesisPluginManager()
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

        self._synth_gates = set(self._synth_gates) - self._basis_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.

        Raises:
            TranspilerError: if a 'method' was specified for the class and is not
                found in the installed plugins list. The list of installed
                plugins can be queried with
                :func:`~qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`
        """
        if self.method not in self.plugins.ext_plugins:
            raise TranspilerError("Specified method: %s not found in plugin list" % self.method)
        # Return fast if we have no synth gates (ie user specified an empty
        # list or the synth gates are all in the basis
        if not self._synth_gates:
            return dag

        default_method = self.plugins.ext_plugins["default"].obj
        plugin_method = self.plugins.ext_plugins[self.method].obj
        if plugin_method.supports_coupling_map:
            dag_bit_indices = {bit: idx for idx, bit in enumerate(dag.qubits)}
        kwargs = {}
        if plugin_method.supports_basis_gates:
            kwargs["basis_gates"] = self._basis_gates
        if plugin_method.supports_natural_direction:
            kwargs["natural_direction"] = self._natural_direction
        if plugin_method.supports_pulse_optimize:
            kwargs["pulse_optimize"] = self._pulse_optimize
        if plugin_method.supports_gate_lengths:
            kwargs["gate_lengths"] = _build_gate_lengths(self._backend_props)
        if plugin_method.supports_gate_errors:
            kwargs["gate_errors"] = _build_gate_errors(self._backend_props)
        supported_bases = plugin_method.supported_bases
        if supported_bases is not None:
            kwargs["matched_basis"] = _choose_bases(self._basis_gates, supported_bases)

        # Handle approximation degree as a special case for backwards compatibility, it's
        # not part of the plugin interface and only something needed for the default
        # pass.
        default_method._approximation_degree = self._approximation_degree
        if self.method == "default":
            plugin_method._approximation_degree = self._approximation_degree

        for node in dag.named_nodes(*self._synth_gates):
            if self._min_qubits is not None and len(node.qargs) < self._min_qubits:
                continue
            if plugin_method.supports_coupling_map:
                kwargs["coupling_map"] = (
                    self._coupling_map,
                    [dag_bit_indices[x] for x in node.qargs],
                )
            synth_dag = None
            unitary = node.op.to_matrix()
            n_qubits = len(node.qargs)
            if (plugin_method.max_qubits is not None and n_qubits > plugin_method.max_qubits) or (
                plugin_method.min_qubits is not None and n_qubits < plugin_method.min_qubits
            ):
                synth_dag = default_method.run(unitary, **kwargs)
            else:
                synth_dag = plugin_method.run(unitary, **kwargs)
            if synth_dag is not None:
                if isinstance(synth_dag, tuple):
                    dag.substitute_node_with_dag(node, synth_dag[0], wires=synth_dag[1])
                else:
                    dag.substitute_node_with_dag(node, synth_dag)
        return dag


def _build_gate_lengths(props):
    gate_lengths = {}
    if props:
        for gate in props._gates:
            gate_lengths[gate] = {}
            for k, v in props._gates[gate].items():
                length = v.get("gate_length")
                if length:
                    gate_lengths[gate][k] = length[0]
            if not gate_lengths[gate]:
                del gate_lengths[gate]
    return gate_lengths


def _build_gate_errors(props):
    gate_errors = {}
    if props:
        for gate in props._gates:
            gate_errors[gate] = {}
            for k, v in props._gates[gate].items():
                error = v.get("gate_error")
                if error:
                    gate_errors[gate][k] = error[0]
            if not gate_errors[gate]:
                del gate_errors[gate]
    return gate_errors


class DefaultUnitarySynthesis(plugin.UnitarySynthesisPlugin):
    """The default unitary synthesis plugin."""

    @property
    def supports_basis_gates(self):
        return True

    @property
    def supports_coupling_map(self):
        return True

    @property
    def supports_natural_direction(self):
        return True

    @property
    def supports_pulse_optimize(self):
        return True

    @property
    def supports_gate_lengths(self):
        return True

    @property
    def supports_gate_errors(self):
        return True

    @property
    def max_qubits(self):
        return None

    @property
    def min_qubits(self):
        return None

    @property
    def supported_bases(self):
        return None

    def run(self, unitary, **options):
        # Approximation degree is set directly as an attribute on the
        # instance by the UnitarySynthesis pass here as it's not part of
        # plugin interface. However if for some reason it's not set assume
        # it's 1.
        approximation_degree = getattr(self, "_approximation_degree", 1)
        basis_gates = options["basis_gates"]
        coupling_map = options["coupling_map"][0]
        natural_direction = options["natural_direction"]
        pulse_optimize = options["pulse_optimize"]
        gate_lengths = options["gate_lengths"]
        gate_errors = options["gate_errors"]
        qubits = options["coupling_map"][1]

        euler_basis = _choose_euler_basis(basis_gates)
        kak_gate = _choose_kak_gate(basis_gates)

        decomposer1q, decomposer2q = None, None
        if euler_basis is not None:
            decomposer1q = one_qubit_decompose.OneQubitEulerDecomposer(euler_basis)
        if kak_gate is not None:
            decomposer2q = TwoQubitBasisDecomposer(
                kak_gate, euler_basis=euler_basis, pulse_optimize=pulse_optimize
            )

        synth_dag = None
        wires = None
        if unitary.shape == (2, 2):
            if decomposer1q is None:
                return None
            synth_dag = circuit_to_dag(decomposer1q._decompose(unitary))
        elif unitary.shape == (4, 4):
            if decomposer2q is None:
                return None
            synth_dag, wires = self._synth_natural_direction(
                unitary,
                coupling_map,
                qubits,
                decomposer2q,
                gate_lengths,
                gate_errors,
                natural_direction,
                approximation_degree,
                pulse_optimize,
            )
        else:
            synth_dag = circuit_to_dag(isometry.Isometry(unitary, 0, 0).definition)

        return synth_dag, wires

    def _synth_natural_direction(
        self,
        su4_mat,
        coupling_map,
        qubits,
        decomposer2q,
        gate_lengths,
        gate_errors,
        natural_direction,
        approximation_degree,
        pulse_optimize,
    ):
        preferred_direction = None
        synth_direction = None
        physical_gate_fidelity = None
        wires = None
        if natural_direction in {None, True} and coupling_map:
            neighbors0 = coupling_map.neighbors(qubits[0])
            zero_one = qubits[1] in neighbors0
            neighbors1 = coupling_map.neighbors(qubits[1])
            one_zero = qubits[0] in neighbors1
            if zero_one and not one_zero:
                preferred_direction = [0, 1]
            if one_zero and not zero_one:
                preferred_direction = [1, 0]
        if (
            natural_direction in {None, True}
            and preferred_direction is None
            and gate_lengths
            and gate_errors
        ):
            len_0_1 = inf
            len_1_0 = inf
            twoq_gate_lengths = gate_lengths.get(decomposer2q.gate.name)
            if twoq_gate_lengths:
                len_0_1 = twoq_gate_lengths.get((qubits[0], qubits[1]), inf)
                len_1_0 = twoq_gate_lengths.get((qubits[1], qubits[0]), inf)
                if len_0_1 < len_1_0:
                    preferred_direction = [0, 1]
                elif len_1_0 < len_0_1:
                    preferred_direction = [1, 0]
                if preferred_direction:
                    twoq_gate_errors = gate_errors.get("cx")
                    gate_error = twoq_gate_errors.get(
                        (qubits[preferred_direction[0]], qubits[preferred_direction[1]])
                    )
                    if gate_error:
                        physical_gate_fidelity = 1 - gate_error
        if natural_direction is True and preferred_direction is None:
            raise TranspilerError(
                f"No preferred direction of gate on qubits {qubits} "
                "could be determined from coupling map or "
                "gate lengths."
            )
        if approximation_degree is not None:
            basis_fidelity = approximation_degree
        else:
            basis_fidelity = physical_gate_fidelity
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
            and pulse_optimize in {True, None}
            and synth_direction != preferred_direction
        ):
            su4_mat_mm = deepcopy(su4_mat)
            su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]]
            su4_mat_mm[:, [1, 2]] = su4_mat_mm[:, [2, 1]]
            synth_dag = circuit_to_dag(decomposer2q(su4_mat_mm, basis_fidelity=basis_fidelity))
            wires = synth_dag.wires[::-1]
        return synth_dag, wires
