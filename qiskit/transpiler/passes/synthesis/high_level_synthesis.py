# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
High-level-synthesis transpiler pass.
"""

from __future__ import annotations

import typing
from typing import Optional, Union, List, Tuple, Callable, Sequence

import numpy as np

from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlledGate, EquivalenceLibrary, equivalence
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)

from .plugin import HighLevelSynthesisPluginManager

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode


class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level objects.

    A higher-level object is an object of type :class:`~.Operation` (e.g., :class:`.Clifford` or
    :class:`.LinearFunction`).  Each object is referred to by its :attr:`~.Operation.name` field
    (e.g., ``"clifford"`` for :class:`.Clifford` objects), and the applicable synthesis methods are
    tied to this name.

    In the config, each method is specified in one of several ways:

    1. a tuple consisting of the name of a known synthesis plugin and a dictionary providing
       additional arguments for the algorithm.
    2. a tuple consisting of an instance of :class:`.HighLevelSynthesisPlugin` and additional
       arguments for the algorithm.
    3. a single string of a known synthesis plugin
    4. a single instance of :class:`.HighLevelSynthesisPlugin`.

    The following example illustrates different ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis plugins should be declared in ``entry-points`` table for
    ``qiskit.synthesis`` in ``pyproject.toml``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the documentation for
    :class:`~.HighLevelSynthesis`.

    For an overview of the complete process of using high-level synthesis, see
    :ref:`using-high-level-synthesis-plugins`.
    """

    def __init__(
        self,
        use_default_on_unspecified: bool = True,
        plugin_selection: str = "sequential",
        plugin_evaluation_fn: Optional[Callable[[QuantumCircuit], int]] = None,
        **kwargs,
    ):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified: if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            plugin_selection: if set to ``"sequential"`` (default), for every higher-level-object
                the synthesis pass will consider the specified methods sequentially, stopping
                at the first method that is able to synthesize the object. If set to ``"all"``,
                all the specified methods will be considered, and the best synthesized circuit,
                according to ``plugin_evaluation_fn`` will be chosen.
            plugin_evaluation_fn: a callable that evaluates the quality of the synthesized
                quantum circuit; a smaller value means a better circuit. If ``None``, the
                quality of the circuit its size (i.e. the number of gates that it contains).
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.plugin_selection = plugin_selection
        self.plugin_evaluation_fn = (
            plugin_evaluation_fn if plugin_evaluation_fn is not None else lambda qc: qc.size()
        )
        self.methods = {}

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


class HighLevelSynthesis(TransformationPass):
    """Synthesize higher-level objects and unroll custom definitions.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`),
    annotated operations (objects of type :class:`.AnnotatedOperation`), and
    custom gates.

    In the most common use-case when either ``basis_gates`` or ``target`` is specified,
    all higher-level objects are synthesized, so the output is a :class:`.DAGCircuit`
    without such objects.
    More precisely, every gate in the output DAG is either directly supported by the target,
    or is in ``equivalence_library``.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).

    The custom gates are synthesized by recursively unrolling their definitions, until every gate
    is either supported by the target or is in the equivalence library.

    When neither ``basis_gates`` nor ``target`` is specified, the pass synthesizes only the top-level
    abstract mathematical objects and annotated operations, without descending into the gate
    ``definitions``. This is consistent with the older behavior of the pass, allowing to synthesize
    some higher-level objects using plugins and leaving the other gates untouched.
    """

    def __init__(
        self,
        hls_config: Optional[HLSConfig] = None,
        coupling_map: Optional[CouplingMap] = None,
        target: Optional[Target] = None,
        use_qubit_indices: bool = False,
        equivalence_library: Optional[EquivalenceLibrary] = None,
        basis_gates: Optional[List[str]] = None,
        min_qubits: int = 0,
    ):
        """
        HighLevelSynthesis initializer.

        Args:
            hls_config: Optional, the high-level-synthesis config that specifies synthesis methods
                and parameters for various high-level-objects in the circuit. If it is not specified,
                the default synthesis methods and parameters will be used.
            coupling_map: Optional, directed graph represented as a coupling map.
            target: Optional, the backend target to use for this pass. If it is specified,
                it will be used instead of the coupling map.
            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after
                the layout is set, that is, whether the qubit indices of higher-level-objects correspond
                to qubit indices on the target backend.
            equivalence_library: The equivalence library used (instructions in this library will not
                be unrolled by this pass).
            basis_gates: Optional, target basis names to unroll to, e.g. `['u3', 'cx']`.
                Ignored if ``target`` is also specified.
            min_qubits: The minimum number of qubits for operations in the input
                dag to translate.
        """
        super().__init__()

        if hls_config is not None:
            self.hls_config = hls_config
        else:
            # When the config file is not provided, we will use the "default" method
            # to synthesize Operations (when available).
            self.hls_config = HLSConfig(True)

        self.hls_plugin_manager = HighLevelSynthesisPluginManager()
        self._coupling_map = coupling_map
        self._target = target
        self._use_qubit_indices = use_qubit_indices
        if target is not None:
            self._coupling_map = self._target.build_coupling_map()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._min_qubits = min_qubits

        self._top_level_only = self._basis_gates is None and self._target is None

        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if not self._top_level_only and (self._target is None or self._target.num_qubits is None):
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            self._device_insts = basic_insts | set(self._basis_gates)
        else:
            self._device_insts = set()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations synthesized.

        Raises:
            TranspilerError: when the transpiler is unable to synthesize the given DAG
            (for instance, when the specified synthesis method is not available).
        """

        # copy dag_op_nodes because we are modifying the DAG below
        dag_op_nodes = dag.op_nodes()

        for node in dag_op_nodes:
            if node.is_control_flow():
                node.op = control_flow.map_blocks(self.run, node.op)
                continue

            if node.is_directive():
                continue

            if dag.has_calibration_for(node) or len(node.qargs) < self._min_qubits:
                continue

            qubits = (
                [dag.find_bit(x).index for x in node.qargs] if self._use_qubit_indices else None
            )

            if self._definitely_skip_node(node, qubits):
                continue

            decomposition, modified = self._recursively_handle_op(node.op, qubits)

            if not modified:
                continue

            if isinstance(decomposition, QuantumCircuit):
                dag.substitute_node_with_dag(
                    node, circuit_to_dag(decomposition, copy_operations=False)
                )
            elif isinstance(decomposition, DAGCircuit):
                dag.substitute_node_with_dag(node, decomposition)
            elif isinstance(decomposition, Operation):
                dag.substitute_node(node, decomposition)

        return dag

    def _definitely_skip_node(self, node: DAGOpNode, qubits: Sequence[int] | None) -> bool:
        """Fast-path determination of whether a node can certainly be skipped (i.e. nothing will
        attempt to synthesise it) without accessing its Python-space `Operation`.

        This is tightly coupled to `_recursively_handle_op`; it exists as a temporary measure to
        avoid Python-space `Operation` creation from a `DAGOpNode` if we wouldn't do anything to the
        node (which is _most_ nodes)."""
        return (
            # The fast path is just for Rust-space standard gates (which excludes
            # `AnnotatedOperation`).
            node.is_standard_gate()
            # If it's a controlled gate, we might choose to do funny things to it.
            and not node.is_controlled_gate()
            # If there are plugins to try, they need to be tried.
            and not self._methods_to_try(node.name)
            # If all the above constraints hold, and it's already supported or the basis translator
            # can handle it, we'll leave it be.
            and (
                self._instruction_supported(node.name, qubits)
                # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
                # `BasisTranslator`, and this is supposed to just be temporary til this is moved
                # into Rust space.
                or (
                    self._equiv_lib is not None
                    and equivalence.Key(name=node.name, num_qubits=node.num_qubits)
                    in self._equiv_lib._key_to_node_index
                )
            )
        )

    def _instruction_supported(self, name: str, qubits: Sequence[int]) -> bool:
        qubits = tuple(qubits) if qubits is not None else None
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if self._target is None or self._target.num_qubits is None:
            return name in self._device_insts
        return self._target.instruction_supported(operation_name=name, qargs=qubits)

    def _recursively_handle_op(
        self, op: Operation, qubits: Optional[List] = None
    ) -> Tuple[Union[QuantumCircuit, DAGCircuit, Operation], bool]:
        """Recursively synthesizes a single operation.

        Note: the reason that this function accepts an operation and not a dag node
        is that it's also used for synthesizing the base operation for an annotated
        gate (i.e. no dag node is available).

        There are several possible results:

        - The given operation is unchanged: e.g., it is supported by the target or is
          in the equivalence library
        - The result is a quantum circuit: e.g., synthesizing Clifford using plugin
        - The result is a DAGCircuit: e.g., when unrolling custom gates
        - The result is an Operation: e.g., adding control to CXGate results in CCXGate
        - The given operation could not be synthesized, raising a transpiler error

        The function returns the result of the synthesis (either a quantum circuit or
        an Operation), and, as an optimization, a boolean indicating whether
        synthesis did anything.

        The function is recursive, for example synthesizing an annotated operation
        involves synthesizing its "base operation" which might also be
        an annotated operation.
        """

        # WARNING: if adding new things in here, ensure that `_definitely_skip_node` is also
        # up-to-date.

        # Try to apply plugin mechanism
        decomposition = self._synthesize_op_using_plugins(op, qubits)
        if decomposition is not None:
            return decomposition, True

        # Handle annotated operations
        decomposition = self._synthesize_annotated_op(op)
        if decomposition:
            return decomposition, True

        # Don't do anything else if processing only top-level
        if self._top_level_only:
            return op, False

        # For non-controlled-gates, check if it's already supported by the target
        # or is in equivalence library
        controlled_gate_open_ctrl = isinstance(op, ControlledGate) and op._open_ctrl
        if not controlled_gate_open_ctrl:
            if self._instruction_supported(op.name, qubits) or (
                self._equiv_lib is not None and self._equiv_lib.has_entry(op)
            ):
                return op, False

        try:
            # extract definition
            definition = op.definition
        except TypeError as err:
            raise TranspilerError(
                f"HighLevelSynthesis was unable to extract definition for {op.name}: {err}"
            ) from err
        except AttributeError:
            # definition is None
            definition = None

        if definition is None:
            raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {op}.")

        dag = circuit_to_dag(definition, copy_operations=False)
        dag = self.run(dag)
        return dag, True

    def _methods_to_try(self, name: str):
        """Get a sequence of methods to try for a given op name."""
        if (methods := self.hls_config.methods.get(name)) is not None:
            # the operation's name appears in the user-provided config,
            # we use the list of methods provided by the user
            return methods
        if (
            self.hls_config.use_default_on_unspecified
            and "default" in self.hls_plugin_manager.method_names(name)
        ):
            # the operation's name does not appear in the user-specified config,
            # we use the "default" method when instructed to do so and the "default"
            # method is available
            return ["default"]
        return []

    def _synthesize_op_using_plugins(
        self, op: Operation, qubits: List
    ) -> Union[QuantumCircuit, None]:
        """
        Attempts to synthesize op using plugin mechanism.
        Returns either the synthesized circuit or None (which occurs when no
        synthesis methods are available or specified).
        """
        hls_plugin_manager = self.hls_plugin_manager

        best_decomposition = None
        best_score = np.inf

        for method in self._methods_to_try(op.name):
            # There are two ways to specify a synthesis method. The more explicit
            # way is to specify it as a tuple consisting of a synthesis algorithm and a
            # list of additional arguments, e.g.,
            #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}), or
            #   ("pmh", {}).
            # When the list of additional arguments is empty, one can also specify
            # just the synthesis algorithm, e.g.,
            #   "pmh".
            if isinstance(method, tuple):
                plugin_specifier, plugin_args = method
            else:
                plugin_specifier = method
                plugin_args = {}

            # There are two ways to specify a synthesis algorithm being run,
            # either by name, e.g. "kms" (which then should be specified in entry_points),
            # or directly as a class inherited from HighLevelSynthesisPlugin (which then
            # does not need to be specified in entry_points).
            if isinstance(plugin_specifier, str):
                if plugin_specifier not in hls_plugin_manager.method_names(op.name):
                    raise TranspilerError(
                        f"Specified method: {plugin_specifier} not found in available "
                        f"plugins for {op.name}"
                    )
                plugin_method = hls_plugin_manager.method(op.name, plugin_specifier)
            else:
                plugin_method = plugin_specifier

            decomposition = plugin_method.run(
                op,
                coupling_map=self._coupling_map,
                target=self._target,
                qubits=qubits,
                **plugin_args,
            )

            # The synthesis methods that are not suited for the given higher-level-object
            # will return None.
            if decomposition is not None:
                if self.hls_config.plugin_selection == "sequential":
                    # In the "sequential" mode the first successful decomposition is
                    # returned.
                    best_decomposition = decomposition
                    break

                # In the "run everything" mode we update the best decomposition
                # discovered
                current_score = self.hls_config.plugin_evaluation_fn(decomposition)
                if current_score < best_score:
                    best_decomposition = decomposition
                    best_score = current_score

        return best_decomposition

    def _synthesize_annotated_op(self, op: Operation) -> Union[Operation, None]:
        """
        Recursively synthesizes annotated operations.
        Returns either the synthesized operation or None (which occurs when the operation
        is not an annotated operation).
        """
        if isinstance(op, AnnotatedOperation):
            # Recursively handle the base operation
            # This results in QuantumCircuit, DAGCircuit or Gate
            synthesized_op, _ = self._recursively_handle_op(op.base_op, qubits=None)

            if isinstance(synthesized_op, AnnotatedOperation):
                raise TranspilerError(
                    "HighLevelSynthesis failed to synthesize the base operation of"
                    " an annotated operation."
                )

            for modifier in op.modifiers:
                # If we have a DAGCircuit at this point, convert it to QuantumCircuit
                if isinstance(synthesized_op, DAGCircuit):
                    synthesized_op = dag_to_circuit(synthesized_op, copy_operations=False)

                if isinstance(modifier, InverseModifier):
                    # Both QuantumCircuit and Gate have inverse method
                    synthesized_op = synthesized_op.inverse()

                elif isinstance(modifier, ControlModifier):
                    # Both QuantumCircuit and Gate have control method, however for circuits
                    # it is more efficient to avoid constructing the controlled quantum circuit.
                    if isinstance(synthesized_op, QuantumCircuit):
                        synthesized_op = synthesized_op.to_gate()

                    synthesized_op = synthesized_op.control(
                        num_ctrl_qubits=modifier.num_ctrl_qubits,
                        label=None,
                        ctrl_state=modifier.ctrl_state,
                        annotated=False,
                    )

                    if isinstance(synthesized_op, AnnotatedOperation):
                        raise TranspilerError(
                            "HighLevelSynthesis failed to synthesize the control modifier."
                        )

                    # Unrolling
                    synthesized_op, _ = self._recursively_handle_op(synthesized_op)

                elif isinstance(modifier, PowerModifier):
                    # QuantumCircuit has power method, and Gate needs to be converted
                    # to a quantum circuit.
                    if isinstance(synthesized_op, QuantumCircuit):
                        qc = synthesized_op
                    else:
                        qc = QuantumCircuit(synthesized_op.num_qubits, synthesized_op.num_clbits)
                        qc.append(
                            synthesized_op,
                            range(synthesized_op.num_qubits),
                            range(synthesized_op.num_clbits),
                        )

                    qc = qc.power(modifier.power)
                    synthesized_op = qc.to_gate()

                    # Unrolling
                    synthesized_op, _ = self._recursively_handle_op(synthesized_op)

                else:
                    raise TranspilerError(f"Unknown modifier {modifier}.")

            return synthesized_op
        return None
