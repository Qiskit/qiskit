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
from functools import partial
from collections.abc import Callable

import numpy as np

from qiskit.circuit.annotated_operation import Modifier
from qiskit.circuit.controlflow.control_flow import ControlFlowOp
from qiskit.circuit.operation import Operation
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlledGate, EquivalenceLibrary, equivalence, Qubit
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

from qiskit._accelerate.high_level_synthesis import QubitTracker, QubitContext
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
        plugin_evaluation_fn: Callable[[QuantumCircuit], int] | None = None,
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


class HLSData:
    """Internal class for keeping immutable data required by HighLevelSynthesis."""

    def __init__(
        self,
        hls_config,
        hls_plugin_manager,
        coupling_map,
        target,
        use_qubit_indices,
        qubits_initially_zero,
        equivalence_library,
        min_qubits,
        top_level_only,
        device_insts,
    ):
        self.hls_config = hls_config
        self.hls_plugin_manager = hls_plugin_manager
        self.coupling_map = coupling_map
        self.target = target
        self.use_qubit_indices = use_qubit_indices
        self.qubits_initially_zero = qubits_initially_zero
        self.equivalence_library = equivalence_library
        self.min_qubits = min_qubits
        self.top_level_only = top_level_only
        self.device_insts = device_insts


class HighLevelSynthesis(TransformationPass):
    r"""Synthesize higher-level objects and unroll custom definitions.

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

    The high-level-synthesis passes information about available auxiliary qubits, and whether their
    state is clean (defined as :math:`|0\rangle`) or dirty (unknown state) to the synthesis routine
    via the respective arguments ``"num_clean_ancillas"`` and ``"num_dirty_ancillas"``.
    If ``qubits_initially_zero`` is ``True`` (default), the qubits are assumed to be in the
    :math:`|0\rangle` state. When appending a synthesized block using auxiliary qubits onto the
    circuit, we first use the clean auxiliary qubits.

    .. note::

        Synthesis methods are assumed to maintain the state of the auxiliary qubits.
        Concretely this means that clean auxiliary qubits must still be in the :math:`|0\rangle`
        state after the synthesized block, while dirty auxiliary qubits are re-used only
        as dirty qubits.

    """

    def __init__(
        self,
        hls_config: HLSConfig | None = None,
        coupling_map: CouplingMap | None = None,
        target: Target | None = None,
        use_qubit_indices: bool = False,
        equivalence_library: EquivalenceLibrary | None = None,
        basis_gates: list[str] | None = None,
        min_qubits: int = 0,
        qubits_initially_zero: bool = True,
    ):
        r"""
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
            qubits_initially_zero: Indicates whether the qubits are initially in the state
                :math:`|0\rangle`. This allows the high-level-synthesis to use clean auxiliary qubits
                (i.e. in the zero state) to synthesize an operation.
        """
        super().__init__()

        # When the config file is not provided, we will use the "default" method
        # to synthesize Operations (when available).
        hls_config = hls_config or HLSConfig(True)
        hls_plugin_manager = HighLevelSynthesisPluginManager()

        if target is not None:
            coupling_map = target.build_coupling_map()
        else:
            coupling_map = coupling_map

        top_level_only = basis_gates is None and target is None

        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if not top_level_only and (target is None or target.num_qubits is None):
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            device_insts = basic_insts | set(basis_gates)
        else:
            device_insts = set()

        self.data = HLSData(
            hls_config=hls_config,
            hls_plugin_manager=hls_plugin_manager,
            coupling_map=coupling_map,
            target=target,
            use_qubit_indices=use_qubit_indices,
            qubits_initially_zero=qubits_initially_zero,
            equivalence_library=equivalence_library,
            min_qubits=min_qubits,
            top_level_only=top_level_only,
            device_insts=device_insts,
        )

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

        # STEP 1: Check if HighLevelSynthesis can be skipped altogether. This is only
        # done at the top-level since this does not update the global qubits tracker.
        for node in dag.op_nodes():
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            if not _definitely_skip_node(self.data, node, qubits, dag):
                break
        else:
            # The for-loop terminates without reaching the break statement
            return dag


        qubits = tuple(dag.find_bit(q).index for q in dag.qubits)
        context = QubitContext(list(range(len(dag.qubits))))
        tracker = QubitTracker(num_qubits=dag.num_qubits())
        if self.data.qubits_initially_zero:
            tracker.set_clean(context.to_globals(qubits))

        # ToDo: try to avoid this conversion
        circuit = dag_to_circuit(dag)
        out_circuit = _run(circuit, self.data, tracker, context, use_ancillas=True)
        assert isinstance(out_circuit, QuantumCircuit)
        out_dag = circuit_to_dag(out_circuit)
        return out_dag


def _run(
    input_circuit: QuantumCircuit,
    data: HLSData,
    tracker: QubitTracker,
    context: QubitContext,
    use_ancillas: bool,
) -> QuantumCircuit:
    """
    The main recursive function that synthesizes a QuantumCircuit.

    Input:
        circuit: the circuit to be synthesized.
        tracker: the global tracker, tracking the state of original qubits.
        context: the correspondence between the circuit's qubits and the global qubits.
        use_ancillas: if True, synthesis algorithms are allowed to use ancillas.

    The function returns the synthesized QuantumCircuit.

    Note that by using the auxiliary qubits to synthesize operations present in the input circuit,
    the synthesized circuit may be defined over more qubits than the input circuit. In this case,
    the function update in-place the global qubits tracker and extends the local-to-global
    context.
    """

    assert isinstance(input_circuit, QuantumCircuit)


    if input_circuit.num_qubits != context.num_qubits():
        raise TranspilerError("HighLevelSynthesis internal error.")

  
    # STEP 2: Analyze the nodes in the circuit. For each node in the circuit that needs
    # to be synthesized, we recursively synthesize it and store the result. For
    # instance, the result of synthesizing a custom gate is a QuantumCircuit corresponding
    # to the (recursively synthesized) gate's definition. When the result is a
    # circuit, we also store its context (the mapping of its qubits to global qubits).
    # In addition, we keep track of the qubit states using the (global) qubits tracker.
    #
    # Note: This is a first version of a potentially more elaborate approach to find
    # good operation/ancilla allocations. The current approach is greedy and just gives
    # all available ancilla qubits to the current operation ("the-first-takes-all" approach).
    # It does not distribute ancilla qubits between different operations present in the circuit.
    synthesized_nodes = {}

    for (idx, inst) in enumerate(input_circuit):
        op = inst.operation
        qubits = tuple(input_circuit.find_bit(q).index for q in inst.qubits)
        processed = False
        synthesized = None
        synthesized_context = None

        # Start by handling special operations. Other cases can also be
        # considered: swaps, automatically simplifying control gate (e.g. if
        # a control is 0).
        if op.name in ["id", "delay", "barrier"]:
            # tracker not updated, these are no-ops
            processed = True

        elif op.name == "reset":
            # reset qubits to 0
            tracker.set_clean(context.to_globals(qubits))
            processed = True

        # check if synthesis for the operation can be skipped
        elif _definitely_skip_op(data, op, qubits, input_circuit):
            tracker.set_dirty(context.to_globals(qubits))

        # next check control flow
        elif isinstance(op, ControlFlowOp):
            # print("I AM HERE")
            # print(f"CONTEXT: {context}")
            # print(f"TRACKER: {tracker}")
            inner_context = context.restrict(qubits)
            # print(f"INNER_CONTEXT: {inner_context}")
            circuit_mapping = partial(
                    _run,
                    data=data,
                    tracker=tracker,
                    context=inner_context,
                    use_ancillas=False,
                )
            synthesized = op.replace_blocks([circuit_mapping(block) for block in op.blocks])
            # print(f"SYNTHESIZED: {synthesized}")
            # synthesized = _wrap_in_circuit(synthesized)
            # synthesized_context = context
            

        # now we are free to synthesize
        else:
            # This returns the synthesized operation and its context (when the result is
            # a circuit, it's the correspondence between its qubits and the global qubits).
            # Also note that the circuit may use auxiliary qubits. The qubits tracker and the
            # current circuit's context are updated in-place.
            synthesized, synthesized_context = _synthesize_operation(
                data, op, qubits, tracker, context, use_ancillas=use_ancillas
            )

        # If the synthesis changed the operation (i.e. it is not None), store the result.
        if synthesized is not None:
            synthesized_nodes[idx] = (synthesized, synthesized_context)

        # If the synthesis did not change anything, just update the qubit tracker.
        elif not processed:
            tracker.set_dirty(context.to_globals(qubits))

    # We did not change anything just return the input.
    if len(synthesized_nodes) == 0:
        if input_circuit.num_qubits != context.num_qubits():
            raise TranspilerError("HighLevelSynthesis internal error.")
        assert isinstance(input_circuit, QuantumCircuit)
        return input_circuit

    # STEP 3. We rebuild the circuit with new operations. Note that we could also
    # check if no operation changed in size and substitute in-place, but rebuilding is
    # generally as fast or faster, unless very few operations are changed.
    out = input_circuit.copy_empty_like()
    num_additional_qubits = context.num_qubits() - out.num_qubits

    if num_additional_qubits > 0:
        out.add_bits([Qubit() for _ in range(num_additional_qubits)])

    index_to_qubit = dict(enumerate(out.qubits))
    outer_to_local = context.to_local_mapping()

    for (idx, inst) in enumerate(input_circuit):
        op = inst.operation

        if op_tuple := synthesized_nodes.get(idx, None):
            op, op_context = op_tuple

            if isinstance(op, Operation):
                # We sgould not be here
                # assert False
                # out.apply_operation_back(op, node.qargs, node.cargs)
                # print(f"{inst.qubits = }, {inst.clbits = }")
                out.append(op, inst.qubits, inst.clbits)
                continue

            assert isinstance(op, QuantumCircuit)

            inner_to_global = op_context.to_global_mapping()
            qubit_map = {
                q: index_to_qubit[outer_to_local[inner_to_global[i]]]
                for (i, q) in enumerate(op.qubits)
            }
            clbit_map = dict(zip(op.clbits, inst.clbits))

            for sub_node in op:
                out.append(
                    sub_node.operation,
                    tuple(qubit_map[qarg] for qarg in sub_node.qubits),
                    tuple(clbit_map[carg] for carg in sub_node.clbits),
                )
            out.global_phase += op.global_phase

            # else:
                # raise TranspilerError(f"Unexpected synthesized type: {type(op)}")
        else:
            out.append(op, inst.qubits, inst.clbits)


    assert isinstance(out, QuantumCircuit)
    if out.num_qubits != context.num_qubits():
        raise TranspilerError("HighLevelSynthesis internal error.")

    return out


def _synthesize_operation(
    data: HLSData,
    operation: Operation,
    qubits: tuple[int],
    tracker: QubitTracker,
    context: QubitContext,
    use_ancillas: bool,
) -> tuple[QuantumCircuit | None, QubitContext | None]:
    """
    Synthesizes an operation. The function receives the qubits on which the operation
    is defined in the current DAG, the correspondence between the qubits of the current
    DAG and the global qubits and the global qubits tracker. The function returns the
    result of synthesizing the operation. The value of `None` means that the operation
    should remain as it is. When it's a circuit, we also return the context, i.e. the
    correspondence of its local qubits and the global qubits. The function changes
    in-place the tracker (state of the global qubits), the qubits (when the synthesized
    operation is defined over additional ancilla qubits), and the context (to keep track
    of where these ancilla qubits maps to).
    """

    synthesized_context = None

    # Try to synthesize the operation. We'll go through the following options:
    #  (1) Annotations: if the operator is annotated, synthesize the base operation
    #       and then apply the modifiers. Returns a circuit (e.g. applying a power)
    #       or operation (e.g adding control on an X gate).
    #  (2) High-level objects: try running the battery of high-level synthesis plugins (e.g.
    #       if the operation is a Clifford). Returns a circuit.
    #  (3) Unrolling custom definitions: try defining the operation if it is not yet
    #       in the set of supported instructions. Returns a circuit.
    #
    # If any of the above were triggered, we will recurse and go again through these steps
    # until no further change occurred. At this point, we convert circuits to DAGs (the final
    # possible return type). If there was no change, we just return ``None``.
    num_original_qubits = len(qubits)
    qubits = list(qubits)

    synthesized = None

    # Try synthesis via HLS -- which will return ``None`` if unsuccessful.
    indices = (
        qubits if data.use_qubit_indices or isinstance(operation, AnnotatedOperation) else None
    )
    if len(hls_methods := _methods_to_try(data, operation.name)) > 0:
        if use_ancillas:
            num_clean_available = tracker.num_clean(context.to_globals(qubits))
            num_dirty_available = tracker.num_dirty(context.to_globals(qubits))
        else:
            num_clean_available = 0
            num_dirty_available = 0

        synthesized = _synthesize_op_using_plugins(
            data,
            hls_methods,
            operation,
            indices,
            num_clean_available,
            num_dirty_available,
            tracker=tracker,
            context=context,
        )

        # It may happen that the plugin synthesis method uses clean/dirty ancilla qubits
        if (synthesized is not None) and (synthesized.num_qubits > len(qubits)):
            # need to borrow more qubits from tracker
            global_aux_qubits = tracker.borrow(
                synthesized.num_qubits - len(qubits), context.to_globals(qubits)
            )
            global_to_local = context.to_local_mapping()

            for aq in global_aux_qubits:
                if aq in global_to_local:
                    qubits.append(global_to_local[aq])
                else:
                    new_local_qubit = context.add_qubit(aq)
                    qubits.append(new_local_qubit)

    # If HLS did not apply, or was unsuccessful, try unrolling custom definitions.
    if synthesized is None and not data.top_level_only:
        synthesized = _get_custom_definition(data, operation, indices)

    if synthesized is None:
        # if we didn't synthesize, there was nothing to unroll
        # updating the tracker will be handled upstream
        pass

    # if it has been synthesized, recurse and finally store the decomposition
    elif isinstance(synthesized, Operation):
        # we should no longer be here!
        assert False

        resynthesized, resynthesized_context = _synthesize_operation(
            data, synthesized, qubits, tracker, context, use_ancillas=use_ancillas
        )

        if resynthesized is not None:
            synthesized = resynthesized
        else:
            tracker.set_dirty(context.to_globals(qubits))
        if isinstance(resynthesized, DAGCircuit):
            synthesized_context = resynthesized_context

    elif isinstance(synthesized, QuantumCircuit):
        # Synthesized is a quantum circuit which we want to process recursively.
        # For example, it's the definition circuit of a custom gate
        # or a circuit obtained by calling a synthesis method on a high-level-object.
        # In the second case, synthesized may have more qubits than the original node.

        as_dag = synthesized
        inner_context = context.restrict(qubits)

        if as_dag.num_qubits != inner_context.num_qubits():
            raise TranspilerError("HighLevelSynthesis internal error.")

        # We save the current state of the tracker to be able to return the ancilla
        # qubits to the current positions. Note that at this point we do not know
        # which ancilla qubits will be allocated.
        saved_tracker = tracker.copy()
        synthesized = _run(
            as_dag, data, tracker, inner_context, use_ancillas=use_ancillas, 
        )
        synthesized_context = inner_context

        if (synthesized is not None) and (synthesized.num_qubits > len(qubits)):
            # need to borrow more qubits from tracker
            global_aux_qubits = tracker.borrow(
                synthesized.num_qubits - len(qubits), context.to_globals(qubits)
            )
            global_to_local = context.to_local_mapping()

            for aq in global_aux_qubits:
                if aq in global_to_local:
                    qubits.append(global_to_local[aq])
                else:
                    new_local_qubit = context.add_qubit(aq)
                    qubits.append(new_local_qubit)

        if len(qubits) > num_original_qubits:
            tracker.replace_state(saved_tracker, context.to_globals(qubits[num_original_qubits:]))

    else:
        raise TranspilerError(f"Unexpected synthesized type: {type(synthesized)}")

    if isinstance(synthesized, DAGCircuit) and synthesized_context is None:
        raise TranspilerError("HighLevelSynthesis internal error.")

    assert synthesized is None or isinstance(synthesized, QuantumCircuit)
    return synthesized, synthesized_context


def _get_custom_definition(
    data: HLSData, inst: Instruction, qubits: list[int] | None
) -> QuantumCircuit | None:
    # check if the operation is already supported natively
    if not (isinstance(inst, ControlledGate) and inst._open_ctrl):
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        inst_supported = _instruction_supported(data, inst.name, qubits)
        if inst_supported or (
            data.equivalence_library is not None and data.equivalence_library.has_entry(inst)
        ):
            return None  # we support this operation already

    # if not, try to get the definition
    try:
        definition = inst.definition
    except (TypeError, AttributeError) as err:
        raise TranspilerError(f"HighLevelSynthesis was unable to define {inst.name}.") from err

    if definition is None:
        raise TranspilerError(f"HighLevelSynthesis was unable to synthesize {inst}.")

    return definition


def _methods_to_try(data: HLSData, name: str):
    """Get a sequence of methods to try for a given op name."""
    if (methods := data.hls_config.methods.get(name)) is not None:
        # the operation's name appears in the user-provided config,
        # we use the list of methods provided by the user
        return methods
    if (
        data.hls_config.use_default_on_unspecified
        and "default" in data.hls_plugin_manager.method_names(name)
    ):
        # the operation's name does not appear in the user-specified config,
        # we use the "default" method when instructed to do so and the "default"
        # method is available
        return ["default"]
    return []


def _synthesize_op_using_plugins(
    data: HLSData,
    hls_methods: list,
    op: Operation,
    qubits: list[int] | None,
    num_clean_ancillas: int = 0,
    num_dirty_ancillas: int = 0,
    tracker: QubitTracker = None,
    context: QubitContext = None,
) -> QuantumCircuit | None:
    """
    Attempts to synthesize op using plugin mechanism.

    The arguments ``num_clean_ancillas`` and ``num_dirty_ancillas`` specify
    the number of clean and dirty qubits available to synthesize the given
    operation. A synthesis method does not need to use these additional qubits.

    Returns either the synthesized circuit or None (which may occur
    when no synthesis methods is available or specified, or when there is
    an insufficient number of auxiliary qubits).
    """
    hls_plugin_manager = data.hls_plugin_manager

    best_decomposition = None
    best_score = np.inf

    for method in hls_methods:
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

        # Set the number of available clean and dirty auxiliary qubits via plugin args.
        plugin_args["num_clean_ancillas"] = num_clean_ancillas
        plugin_args["num_dirty_ancillas"] = num_dirty_ancillas
        plugin_args["_qubit_tracker"] = tracker
        plugin_args["_qubit_context"] = context
        plugin_args["_data"] = data

        decomposition = plugin_method.run(
            op,
            coupling_map=data.coupling_map,
            target=data.target,
            qubits=qubits,
            **plugin_args,
        )

        # The synthesis methods that are not suited for the given higher-level-object
        # will return None.
        if decomposition is not None:
            if data.hls_config.plugin_selection == "sequential":
                # In the "sequential" mode the first successful decomposition is
                # returned.
                best_decomposition = decomposition
                break

            # In the "run everything" mode we update the best decomposition
            # discovered
            current_score = data.hls_config.plugin_evaluation_fn(decomposition)
            if current_score < best_score:
                best_decomposition = decomposition
                best_score = current_score

    return best_decomposition


def _definitely_skip_node(
    data: HLSData, node: DAGOpNode, qubits: tuple[int] | None, dag: DAGCircuit
) -> bool:
    """Fast-path determination of whether a node can certainly be skipped (i.e. nothing will
    attempt to synthesise it) without accessing its Python-space `Operation`.

    This is tightly coupled to `_recursively_handle_op`; it exists as a temporary measure to
    avoid Python-space `Operation` creation from a `DAGOpNode` if we wouldn't do anything to the
    node (which is _most_ nodes)."""

    if (
        dag._has_calibration_for(node)
        or len(node.qargs) < data.min_qubits
        or node.is_directive()
        or (_instruction_supported(data, node.name, qubits) and not node.is_control_flow())
    ):
        return True

    return (
        # The fast path is just for Rust-space standard gates (which excludes
        # `AnnotatedOperation`).
        node.is_standard_gate()
        # We don't have the fast-path for controlled gates over 3 or more qubits.
        # However, we most probably want the fast-path for controlled 2-qubit gates
        # (such as CX, CZ, CY, CH, CRX, and so on), so "_definitely_skip_node" should
        # not immediately return False when encountering a controlled gate over 2 qubits.
        and not (node.is_controlled_gate() and node.num_qubits >= 3)
        # If there are plugins to try, they need to be tried.
        and not _methods_to_try(data, node.name)
        # If all the above constraints hold, and it's already supported or the basis translator
        # can handle it, we'll leave it be.
        and (
            # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
            # `BasisTranslator`, and this is supposed to just be temporary til this is moved
            # into Rust space.
            data.equivalence_library is not None
            and equivalence.Key(name=node.name, num_qubits=node.num_qubits)
            in data.equivalence_library.keys()
        )
    )


# ToDo: try to avoid duplication with other function
def _definitely_skip_op(
    data: HLSData, op: Operation, qubits: tuple[int], dag: DAGCircuit
) -> bool:
    """Fast-path determination of whether a node can certainly be skipped (i.e. nothing will
    attempt to synthesise it) without accessing its Python-space `Operation`.

    This is tightly coupled to `_recursively_handle_op`; it exists as a temporary measure to
    avoid Python-space `Operation` creation from a `DAGOpNode` if we wouldn't do anything to the
    node (which is _most_ nodes)."""

    assert qubits is not None

    if (
        len(qubits) < data.min_qubits
        or getattr(op, "_directive", False)
        or (_instruction_supported(data, op.name, qubits) and not isinstance(op, ControlFlowOp))
    ):
        return True

    return (
        # The fast path is just for Rust-space standard gates (which excludes
        # `AnnotatedOperation`).
        getattr(op, "_standard_gate", False)
        # We don't have the fast-path for controlled gates over 3 or more qubits.
        # However, we most probably want the fast-path for controlled 2-qubit gates
        # (such as CX, CZ, CY, CH, CRX, and so on), so "_definitely_skip_node" should
        # not immediately return False when encountering a controlled gate over 2 qubits.
        and not (isinstance(op, ControlFlowOp) and op.num_qubits >= 3)
        # If there are plugins to try, they need to be tried.
        and not _methods_to_try(data, op.name)
        # If all the above constraints hold, and it's already supported or the basis translator
        # can handle it, we'll leave it be.
        and (
            # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
            # `BasisTranslator`, and this is supposed to just be temporary til this is moved
            # into Rust space.
            data.equivalence_library is not None
            and equivalence.Key(name=op.name, num_qubits=len(qubits))
            in data.equivalence_library.keys()
        )
    )


def _instruction_supported(data: HLSData, name: str, qubits: tuple[int] | None) -> bool:
    # include path for when target exists but target.num_qubits is None (BasicSimulator)
    if data.target is None or data.target.num_qubits is None:
        return name in data.device_insts
    return data.target.instruction_supported(operation_name=name, qargs=qubits)


def _wrap_in_circuit(op: Operation) -> QuantumCircuit:
    circuit = QuantumCircuit(op.num_qubits, op.num_clbits)
    circuit.append(op, circuit.qubits, circuit.clbits)
    return circuit