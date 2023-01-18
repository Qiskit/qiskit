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
from itertools import product

from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info.synthesis import one_qubit_decompose
from qiskit.quantum_info.synthesis.xx_decompose import XXDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.library.standard_gates import (
    iSwapGate,
    CXGate,
    CZGate,
    RXXGate,
    RZXGate,
    ECRGate,
)
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.utils import control_flow
from qiskit.providers.models import BackendProperties


KAK_GATE_NAMES = {
    "cx": CXGate(),
    "cz": CZGate(),
    "iswap": iSwapGate(),
    "rxx": RXXGate(pi / 2),
    "ecr": ECRGate(),
    "rzx": RZXGate(pi / 4),  # typically pi/6 is also available
}


def _choose_kak_gate(basis_gates):
    """Choose the first available 2q gate to use in the KAK decomposition."""
    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(KAK_GATE_NAMES.keys())
    if kak_gates:
        kak_gate = KAK_GATE_NAMES[kak_gates.pop()]

    return kak_gate


def _find_matching_kak_gates(target):
    """Return list of available 2q gates to use in the KAK decomposition."""
    kak_gates = []
    for name in target:
        if name in KAK_GATE_NAMES:
            kak_gates.append(KAK_GATE_NAMES[name])
            continue
        op = target.operation_from_name(name)
        if isinstance(op, RXXGate) and (
            isinstance(op.params[0], Parameter) or op.params[0] == pi / 2
        ):
            kak_gates.append((KAK_GATE_NAMES["rxx"], name))
        elif isinstance(op, RZXGate) and (
            isinstance(op.params[0], Parameter) or op.params[0] == pi / 4
        ):
            kak_gates.append((KAK_GATE_NAMES["rzx"], name))
    return kak_gates


def _choose_euler_basis(basis_gates):
    """Choose the first available 1q basis to use in the Euler decomposition."""
    basis_set = set(basis_gates or [])

    for basis, gates in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():

        if set(gates).issubset(basis_set):
            return basis

    return None


def _find_matching_euler_bases(target):
    """Find matching availablee 1q basis to use in the Euler decomposition."""
    euler_basis_gates = []
    basis_set = target.keys()
    for basis, gates in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            euler_basis_gates.append(basis)
    return euler_basis_gates


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


def _basis_gates_to_decomposer_2q(basis_gates, pulse_optimize=None):
    kak_gate = _choose_kak_gate(basis_gates)
    euler_basis = _choose_euler_basis(basis_gates)

    if isinstance(kak_gate, RZXGate):
        backup_optimizer = TwoQubitBasisDecomposer(
            CXGate(), euler_basis=euler_basis, pulse_optimize=pulse_optimize
        )
        return XXDecomposer(euler_basis=euler_basis, backup_optimizer=backup_optimizer)
    elif kak_gate is not None:
        return TwoQubitBasisDecomposer(
            kak_gate, euler_basis=euler_basis, pulse_optimize=pulse_optimize
        )
    else:
        return None


class UnitarySynthesis(TransformationPass):
    """Synthesize gates according to their basis gates."""

    def __init__(
        self,
        basis_gates: List[str] = None,
        approximation_degree: float = 1,
        coupling_map: CouplingMap = None,
        backend_props: BackendProperties = None,
        pulse_optimize: Union[bool, None] = None,
        natural_direction: Union[bool, None] = None,
        synth_gates: Union[List[str], None] = None,
        method: str = "default",
        min_qubits: int = None,
        plugin_config: dict = None,
        target: Target = None,
    ):
        """Synthesize unitaries over some basis gates.

        This pass can approximate 2-qubit unitaries given some
        approximation closeness measure (expressed as
        ``approximation_degree``). Other unitaries are synthesized
        exactly.

        Args:
            basis_gates (list[str]): List of gate names to target. If this is
                not specified the ``target`` argument must be used. If both this
                and the ``target`` are specified the value of ``target`` will
                be used and this will be ignored.
            approximation_degree (float): Closeness of approximation
                (0: lowest, 1: highest).
            coupling_map (CouplingMap): the coupling map of the backend
                in case synthesis is done on a physical circuit. The
                directionality of the coupling_map will be taken into
                account if ``pulse_optimize`` is ``True``/``None`` and ``natural_direction``
                is ``True``/``None``.
            backend_props (BackendProperties): Properties of a backend to
                synthesize for (e.g. gate fidelities).
            pulse_optimize (bool): Whether to optimize pulses during
                synthesis. A value of ``None`` will attempt it but fall
                back if it does not succeed. A value of ``True`` will raise
                an error if pulse-optimized synthesis does not succeed.
            natural_direction (bool): Whether to apply synthesis considering
                directionality of 2-qubit gates. Only applies when
                ``pulse_optimize`` is ``True`` or ``None``. The natural direction is
                determined by first checking to see whether the
                coupling map is unidirectional.  If there is no
                coupling map or the coupling map is bidirectional,
                the gate direction with the shorter
                duration from the backend properties will be used. If
                set to True, and a natural direction can not be
                determined, raises :class:`~TranspileError`. If set to None, no
                exception will be raised if a natural direction can
                not be determined.
            synth_gates (list[str]): List of gates to synthesize. If None and
                ``pulse_optimize`` is False or None, default to
                ``['unitary']``. If ``None`` and ``pulse_optimize == True``,
                default to ``['unitary', 'swap']``
            method (str): The unitary synthesis method plugin to use.
            min_qubits: The minimum number of qubits in the unitary to synthesize. If this is set
                and the unitary is less than the specified number of qubits it will not be
                synthesized.
            plugin_config: Optional extra configuration arguments (as a ``dict``)
                which are passed directly to the specified unitary synthesis
                plugin. By default, this will have no effect as the default
                plugin has no extra arguments. Refer to the documentation of
                your unitary synthesis plugin on how to use this.
            target: The optional :class:`~.Target` for the target device the pass
                is compiling for. If specified this will supersede the values
                set for ``basis_gates``, ``coupling_map``, and ``backend_props``.
        """
        super().__init__()
        self._basis_gates = set(basis_gates or ())
        self._approximation_degree = approximation_degree
        self._min_qubits = min_qubits
        self.method = method
        self.plugins = None
        if method != "default":
            self.plugins = plugin.UnitarySynthesisPluginManager()
        self._coupling_map = coupling_map
        self._backend_props = backend_props
        self._pulse_optimize = pulse_optimize
        self._natural_direction = natural_direction
        self._plugin_config = plugin_config
        self._target = target
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        if synth_gates:
            self._synth_gates = synth_gates
        else:
            if pulse_optimize:
                self._synth_gates = ["unitary", "swap"]
            else:
                self._synth_gates = ["unitary"]

        self._synth_gates = set(self._synth_gates) - self._basis_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the UnitarySynthesis pass on ``dag``.

        Args:
            dag: input dag.

        Returns:
            Output dag with UnitaryGates synthesized to target basis.

        Raises:
            TranspilerError: if ``method`` was specified for the class and is not
                found in the installed plugins list. The list of installed
                plugins can be queried with
                :func:`~qiskit.transpiler.passes.synthesis.plugin.unitary_synthesis_plugin_names`
        """
        if self.method != "default" and self.method not in self.plugins.ext_plugins:
            raise TranspilerError("Specified method: %s not found in plugin list" % self.method)
        # Return fast if we have no synth gates (ie user specified an empty
        # list or the synth gates are all in the basis
        if not self._synth_gates:
            return dag
        if self.plugins:
            plugin_method = self.plugins.ext_plugins[self.method].obj
        else:
            plugin_method = DefaultUnitarySynthesis()
        plugin_kwargs = {"config": self._plugin_config}
        _gate_lengths = _gate_errors = None

        if self.method == "default":
            # If the method is the default, we only need to evaluate one set of keyword arguments.
            # To simplify later logic, and avoid cases where static analysis might complain that we
            # haven't initialised the "default" handler, we rebind the names so they point to the
            # same object as the chosen method.
            default_method = plugin_method
            default_kwargs = plugin_kwargs
            method_list = [(plugin_method, plugin_kwargs)]
        else:
            # If the method is not the default, we still need to initialise the default plugin's
            # keyword arguments in case we have to fall back on it during the actual run.
            default_method = self.plugins.ext_plugins["default"].obj
            default_kwargs = {}
            method_list = [(plugin_method, plugin_kwargs), (default_method, default_kwargs)]

        for method, kwargs in method_list:
            if method.supports_basis_gates:
                kwargs["basis_gates"] = self._basis_gates
            if method.supports_natural_direction:
                kwargs["natural_direction"] = self._natural_direction
            if method.supports_pulse_optimize:
                kwargs["pulse_optimize"] = self._pulse_optimize
            if method.supports_gate_lengths:
                _gate_lengths = _gate_lengths or _build_gate_lengths(
                    self._backend_props, self._target
                )
                kwargs["gate_lengths"] = _gate_lengths
            if method.supports_gate_errors:
                _gate_errors = _gate_errors or _build_gate_errors(self._backend_props, self._target)
                kwargs["gate_errors"] = _gate_errors
            supported_bases = method.supported_bases
            if supported_bases is not None:
                kwargs["matched_basis"] = _choose_bases(self._basis_gates, supported_bases)
            if method.supports_target:
                kwargs["target"] = self._target

        # Handle approximation degree as a special case for backwards compatibility, it's
        # not part of the plugin interface and only something needed for the default
        # pass.
        # pylint: disable=attribute-defined-outside-init
        default_method._approximation_degree = self._approximation_degree
        if self.method == "default":
            # pylint: disable=attribute-defined-outside-init
            plugin_method._approximation_degree = self._approximation_degree
        return self._run_main_loop(
            dag, plugin_method, plugin_kwargs, default_method, default_kwargs
        )

    def _run_main_loop(self, dag, plugin_method, plugin_kwargs, default_method, default_kwargs):
        """Inner loop for the optimizer, after all DAG-independent set-up has been completed."""

        def _recurse(dag):
            # This isn't quite a trivially recursive call because we need to close over the
            # arguments to the function.  The loop is sufficiently long that it's cleaner to do it
            # in a separate method rather than define a helper closure within `self.run`.
            return self._run_main_loop(
                dag, plugin_method, plugin_kwargs, default_method, default_kwargs
            )

        for node in dag.op_nodes(ControlFlowOp):
            node.op = control_flow.map_blocks(_recurse, node.op)

        dag_bit_indices = (
            {bit: i for i, bit in enumerate(dag.qubits)}
            if plugin_method.supports_coupling_map or default_method.supports_coupling_map
            else {}
        )

        for node in dag.named_nodes(*self._synth_gates):
            if self._min_qubits is not None and len(node.qargs) < self._min_qubits:
                continue
            synth_dag = None
            unitary = node.op.to_matrix()
            n_qubits = len(node.qargs)
            if (plugin_method.max_qubits is not None and n_qubits > plugin_method.max_qubits) or (
                plugin_method.min_qubits is not None and n_qubits < plugin_method.min_qubits
            ):
                method, kwargs = default_method, default_kwargs
            else:
                method, kwargs = plugin_method, plugin_kwargs
            if method.supports_coupling_map:
                kwargs["coupling_map"] = (
                    self._coupling_map,
                    [dag_bit_indices[x] for x in node.qargs],
                )
            synth_dag = method.run(unitary, **kwargs)
            if synth_dag is not None:
                if isinstance(synth_dag, tuple):
                    dag.substitute_node_with_dag(node, synth_dag[0], wires=synth_dag[1])
                else:
                    dag.substitute_node_with_dag(node, synth_dag)
        return dag


def _build_gate_lengths(props=None, target=None):
    gate_lengths = {}
    if target is not None:
        for gate, prop_dict in target.items():
            gate_lengths[gate] = {}
            for qubit, gate_props in prop_dict.items():
                if gate_props is not None and gate_props.duration is not None:
                    gate_lengths[gate][qubit] = gate_props.duration
    elif props is not None:
        for gate in props._gates:
            gate_lengths[gate] = {}
            for k, v in props._gates[gate].items():
                length = v.get("gate_length")
                if length:
                    gate_lengths[gate][k] = length[0]
            if not gate_lengths[gate]:
                del gate_lengths[gate]
    return gate_lengths


def _build_gate_errors(props=None, target=None):
    gate_errors = {}
    if target is not None:
        for gate, prop_dict in target.items():
            gate_errors[gate] = {}
            for qubit, gate_props in prop_dict.items():
                if gate_props is not None and gate_props.error is not None:
                    gate_errors[gate][qubit] = gate_props.error
    if props is not None:
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

    @property
    def supports_target(self):
        return True

    def __init__(self):
        super().__init__()
        self._decomposer_cache = {}

    def _find_decomposer_2q_from_target(self, target, qubits, pulse_optimize):
        qubits_tuple = tuple(qubits)
        reverse_tuple = (qubits[1], qubits[0])
        if qubits_tuple in self._decomposer_cache:
            return self._decomposer_cache[qubits_tuple]

        matching = {}
        reverse = {}
        kak_gates = _find_matching_kak_gates(target)
        euler_basis_gates = _find_matching_euler_bases(target)
        decomposers_2q = []
        # find all decomposers
        for kak_gate, euler_basis in product(kak_gates, euler_basis_gates):
            gate_name = None
            if isinstance(kak_gate, tuple):
                gate_name = kak_gate[1]
                kak_gate = kak_gate[0]
            if isinstance(kak_gate, RZXGate):
                backup_optimizer = TwoQubitBasisDecomposer(
                    CXGate(), euler_basis=euler_basis, pulse_optimize=pulse_optimize
                )
                decomposer = XXDecomposer(
                    euler_basis=euler_basis, backup_optimizer=backup_optimizer
                )
                if gate_name is not None:
                    decomposer.gate_name = gate_name
                decomposers_2q.append(decomposer)
            elif kak_gate is not None:
                decomposer = TwoQubitBasisDecomposer(
                    kak_gate, euler_basis=euler_basis, pulse_optimize=pulse_optimize
                )
                if gate_name is not None:
                    decomposer.gate_name = gate_name
                decomposers_2q.append(decomposer)

        # Find lowest error matching or reverse decomposer and use that
        for index, decomposer in enumerate(decomposers_2q):
            gate_name = getattr(decomposer, "gate_name", decomposer.gate.name)
            props_dict = target[gate_name]
            if target.instruction_supported(gate_name, qubits_tuple):
                if props_dict is None or None in props_dict:
                    error = 0.0
                else:
                    error = getattr(props_dict[qubits_tuple], "error", 0.0)
                    if error is None:
                        error = 0.0
                matching[index] = error
            # Skip reverse check if we already have matching
            elif not matching and target.instruction_supported(gate_name, reverse_tuple):
                if props_dict is None or None in props_dict:
                    error = 0.0
                else:
                    error = getattr(props_dict[reverse_tuple], "error", 0.0)
                    if error is None:
                        error = 0.0
                reverse[index] = error
        preferred_direction = None
        if matching:
            preferred_direction = [0, 1]
            min_error_index = min(matching, key=matching.get)
            decomposer2q = decomposers_2q[min_error_index]
        elif reverse:
            preferred_direction = [1, 0]
            min_error_index = min(reverse, key=reverse.get)
            decomposer2q = decomposers_2q[min_error_index]
        # If no matching or reverse direction is found just pick one, if natural direction is
        # enforced it will fail later
        else:
            decomposer2q = decomposers_2q[0]
        self._decomposer_cache[qubits_tuple] = (decomposer2q, preferred_direction)
        return (decomposer2q, preferred_direction)

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
        target = options["target"]

        synth_dag = None
        wires = None
        if unitary.shape == (2, 2):
            if target is not None:
                euler_basis = _choose_euler_basis(target.operation_names_for_qargs(tuple(qubits)))
            else:
                euler_basis = _choose_euler_basis(basis_gates)
            if euler_basis is not None:
                decomposer1q = one_qubit_decompose.OneQubitEulerDecomposer(euler_basis)
            else:
                decomposer1q = None
            if decomposer1q is None:
                return None
            synth_dag = circuit_to_dag(decomposer1q._decompose(unitary))
        elif unitary.shape == (4, 4):
            preferred_direction = None
            if target is not None:
                decomposer2q, preferred_direction = self._find_decomposer_2q_from_target(
                    target, qubits, pulse_optimize
                )
            else:
                decomposer2q = _basis_gates_to_decomposer_2q(
                    basis_gates, pulse_optimize=pulse_optimize
                )
            if not decomposer2q:
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
                target,
                preferred_direction,
            )
        else:
            from qiskit.quantum_info.synthesis.qsd import (  # pylint: disable=cyclic-import
                qs_decomposition,
            )

            synth_dag = circuit_to_dag(qs_decomposition(unitary))

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
        target,
        preferred_direction=None,
    ):
        synth_direction = None
        physical_gate_fidelity = None
        wires = None
        if natural_direction in {None, True} and (
            coupling_map or (target is not None and decomposer2q and not preferred_direction)
        ):
            if coupling_map is not None:
                cmap = coupling_map
            else:
                cmap = target.build_coupling_map()
            # If we don't have a defined coupling map (either from the input)
            # or from the target we can't check for a natural direction
            if cmap is not None:
                neighbors0 = cmap.neighbors(qubits[0])
                zero_one = qubits[1] in neighbors0
                neighbors1 = cmap.neighbors(qubits[1])
                one_zero = qubits[0] in neighbors1
                if zero_one and not one_zero:
                    preferred_direction = [0, 1]
                if one_zero and not zero_one:
                    preferred_direction = [1, 0]
        if (
            natural_direction in {None, True}
            and preferred_direction is None
            and (gate_lengths and gate_errors)
        ):
            len_0_1 = inf
            len_1_0 = inf
            gate_name = getattr(decomposer2q, "gate_name", decomposer2q.gate.name)
            twoq_gate_lengths = gate_lengths.get(gate_name)
            if twoq_gate_lengths:
                len_0_1 = twoq_gate_lengths.get((qubits[0], qubits[1]), inf)
                len_1_0 = twoq_gate_lengths.get((qubits[1], qubits[0]), inf)
            if len_0_1 < len_1_0:
                preferred_direction = [0, 1]
            elif len_1_0 < len_0_1:
                preferred_direction = [1, 0]
            if preferred_direction:
                twoq_gate_errors = gate_errors.get(gate_name)
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
        if not isinstance(decomposer2q, XXDecomposer):
            synth_circ = decomposer2q(su4_mat, basis_fidelity=basis_fidelity)
        else:
            synth_circ = decomposer2q(su4_mat)
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
