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

# pylint: disable=missing-function-docstring

"""Translates gates to a target basis using a given equivalence library."""

import time
import logging

from itertools import zip_longest
from collections import defaultdict
from functools import singledispatch

import retworkx

from qiskit.circuit import Gate, ParameterVector, QuantumRegister, ControlFlowOp, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


logger = logging.getLogger(__name__)


class BasisTranslator(TransformationPass):
    """Translates gates to a target basis by searching for a set of translations
    from a given EquivalenceLibrary.

    This pass operates in several steps:

    * Determine the source basis from the input circuit.
    * Perform a Dijkstra search over basis sets, starting from the device's
      target_basis new gates are being generated using the rules from the provided
      EquivalenceLibrary and the search stops if all gates in the source basis have
      been generated.
    * The found path, as a set of rules from the EquivalenceLibrary, is composed
      into a set of gate replacement rules.
    * The composed replacement rules are applied in-place to each op node which
      is not already in the target_basis.

    If the target keyword argument is specified and that
    :class:`~qiskit.transpiler.Target` objects contains operations
    which are non-global (i.e. they are defined only for a subset of qubits),
    as calculated by :meth:`~qiskit.transpiler.Target.get_non_global_operation_names`,
    this pass will attempt to match the output translation to those constraints.
    For 1 qubit operations this is straightforward, the pass will perform a
    search using the union of the set of global operations with the set of operations
    defined solely on that qubit. For multi-qubit gates this is a bit more involved,
    while the behavior is initially similar to the single qubit case, just using all
    the qubits the operation is run on (where order is not significant) isn't sufficient.
    We also need to consider any potential local qubits defined on subsets of the
    quantum arguments for the multi-qubit operation. This means the target used for the
    search of a non-global multi-qubit gate is the union of global operations, non-global
    multi-qubit gates sharing the same qubits, and any non-global gates defined on
    any subset of the qubits used.


    .. note::

        In the case of non-global operations it is possible for a single
        execution of this pass to output an incomplete translation if any
        non-global gates are defined on qubits that are a subset of a larger
        multi-qubit gate. For example, if you have a ``u`` gate only defined on
        qubit 0 and an ``x`` gate only on qubit 1 it is possible when
        translating a 2 qubit operation on qubit 0 and 1 that the output might
        have ``u`` on qubit 1 and ``x`` on qubit 0. Typically running this pass
        a second time will correct these issues.
    """

    def __init__(self, equivalence_library, target_basis, target=None):
        """Initialize a BasisTranslator instance.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            target_basis (list[str]): Target basis names to unroll to, e.g. `['u3', 'cx']`.
            target (Target): The backend compilation target
        """

        super().__init__()
        self._equiv_lib = equivalence_library
        self._target_basis = target_basis
        self._target = target
        self._non_global_operations = None
        self._qargs_with_non_global_operation = {}  # pylint: disable=invalid-name
        if target is not None:
            self._non_global_operations = self._target.get_non_global_operation_names()
            self._qargs_with_non_global_operation = defaultdict(set)
            for gate in self._non_global_operations:
                for qarg in self._target[gate]:
                    self._qargs_with_non_global_operation[qarg].add(gate)

    def run(self, dag):
        """Translate an input DAGCircuit to the target basis.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            TranspilerError: if the target basis cannot be reached

        Returns:
            DAGCircuit: translated circuit.
        """
        if self._target_basis is None and self._target is None:
            return dag

        qarg_indices = {qubit: index for index, qubit in enumerate(dag.qubits)}
        # Names of instructions assumed to supported by any backend.
        if self._target is None:
            basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay"]
            target_basis = set(self._target_basis)
            source_basis = set(_extract_basis(dag))
            qargs_local_source_basis = {}
        else:
            basic_instrs = ["barrier", "snapshot"]
            target_basis = self._target.keys() - set(self._non_global_operations)
            source_basis, qargs_local_source_basis = self._extract_basis_target(dag, qarg_indices)

        target_basis = set(target_basis).union(basic_instrs)

        logger.info(
            "Begin BasisTranslator from source basis %s to target basis %s.",
            source_basis,
            target_basis,
        )

        # Search for a path from source to target basis.
        search_start_time = time.time()
        basis_transforms = _basis_search(self._equiv_lib, source_basis, target_basis)

        qarg_local_basis_transforms = {}
        for qarg, local_source_basis in qargs_local_source_basis.items():
            expanded_target = set(target_basis)
            # For any multiqubit operation that contains a subset of qubits that
            # has a non-local operation, include that non-local operation in the
            # search. This matches with the check we did above to include those
            # subset non-local operations in the check here.
            if len(qarg) > 1:
                for non_local_qarg, local_basis in self._qargs_with_non_global_operation.items():
                    if qarg.issuperset(non_local_qarg):
                        expanded_target |= local_basis
            else:
                expanded_target |= self._qargs_with_non_global_operation[tuple(qarg)]

            logger.info(
                "Performing BasisTranslator search from source basis %s to target "
                "basis %s on qarg %s.",
                local_source_basis,
                expanded_target,
                qarg,
            )
            local_basis_transforms = _basis_search(
                self._equiv_lib, local_source_basis, expanded_target
            )

            if local_basis_transforms is None:
                raise TranspilerError(
                    "Unable to map source basis {} to target basis {} on qarg {} "
                    "over library {}.".format(
                        local_source_basis, expanded_target, qarg, self._equiv_lib
                    )
                )

            qarg_local_basis_transforms[qarg] = local_basis_transforms

        search_end_time = time.time()
        logger.info(
            "Basis translation path search completed in %.3fs.", search_end_time - search_start_time
        )

        if basis_transforms is None:
            raise TranspilerError(
                "Unable to map source basis {} to target basis {} "
                "over library {}.".format(source_basis, target_basis, self._equiv_lib)
            )

        # Compose found path into a set of instruction substitution rules.

        compose_start_time = time.time()
        instr_map = _compose_transforms(basis_transforms, source_basis, dag)
        extra_instr_map = {
            qarg: _compose_transforms(transforms, qargs_local_source_basis[qarg], dag)
            for qarg, transforms in qarg_local_basis_transforms.items()
        }

        compose_end_time = time.time()
        logger.info(
            "Basis translation paths composed in %.3fs.", compose_end_time - compose_start_time
        )

        # Replace source instructions with target translations.

        replace_start_time = time.time()

        def apply_translation(dag, wire_map):
            dag_updated = False
            for node in dag.op_nodes():
                node_qargs = tuple(wire_map[bit] for bit in node.qargs)
                qubit_set = frozenset(node_qargs)
                if node.name in target_basis:
                    if isinstance(node.op, ControlFlowOp):
                        flow_blocks = []
                        for block in node.op.blocks:
                            dag_block = circuit_to_dag(block)
                            dag_updated = apply_translation(
                                dag_block,
                                {
                                    inner: wire_map[outer]
                                    for inner, outer in zip(block.qubits, node.qargs)
                                },
                            )
                            if dag_updated:
                                flow_circ_block = dag_to_circuit(dag_block)
                            else:
                                flow_circ_block = block
                            flow_blocks.append(flow_circ_block)
                        node.op = node.op.replace_blocks(flow_blocks)
                    continue
                if (
                    node_qargs in self._qargs_with_non_global_operation
                    and node.name in self._qargs_with_non_global_operation[node_qargs]
                ):
                    continue

                if dag.has_calibration_for(node):
                    continue
                if qubit_set in extra_instr_map:
                    self._replace_node(dag, node, extra_instr_map[qubit_set])
                elif (node.op.name, node.op.num_qubits) in instr_map:
                    self._replace_node(dag, node, instr_map)
                else:
                    raise TranspilerError(f"BasisTranslator did not map {node.name}.")
                dag_updated = True
            return dag_updated

        apply_translation(dag, qarg_indices)
        replace_end_time = time.time()
        logger.info(
            "Basis translation instructions replaced in %.3fs.",
            replace_end_time - replace_start_time,
        )

        return dag

    def _replace_node(self, dag, node, instr_map):
        target_params, target_dag = instr_map[node.op.name, node.op.num_qubits]
        if len(node.op.params) != len(target_params):
            raise TranspilerError(
                "Translation num_params not equal to op num_params."
                "Op: {} {} Translation: {}\n{}".format(
                    node.op.params, node.op.name, target_params, target_dag
                )
            )

        if node.op.params:
            # Convert target to circ and back to assign_parameters, since
            # DAGCircuits won't have a ParameterTable.
            target_circuit = dag_to_circuit(target_dag)

            target_circuit.assign_parameters(
                dict(zip_longest(target_params, node.op.params)), inplace=True
            )

            bound_target_dag = circuit_to_dag(target_circuit)
        else:
            bound_target_dag = target_dag

        if len(bound_target_dag.op_nodes()) == 1 and len(
            bound_target_dag.op_nodes()[0].qargs
        ) == len(node.qargs):
            dag_op = bound_target_dag.op_nodes()[0].op
            # dag_op may be the same instance as other ops in the dag,
            # so if there is a condition, need to copy
            if getattr(node.op, "condition", None):
                dag_op = dag_op.copy()
            dag.substitute_node(node, dag_op, inplace=True)

            if bound_target_dag.global_phase:
                dag.global_phase += bound_target_dag.global_phase
        else:
            dag.substitute_node_with_dag(node, bound_target_dag)

    def _extract_basis_target(
        self, dag, qarg_indices, source_basis=None, qargs_local_source_basis=None
    ):
        if source_basis is None:
            source_basis = set()
        if qargs_local_source_basis is None:
            qargs_local_source_basis = defaultdict(set)
        for node in dag.op_nodes():
            qargs = tuple(qarg_indices[bit] for bit in node.qargs)
            if dag.has_calibration_for(node):
                continue
            # Treat the instruction as on an incomplete basis if the qargs are in the
            # qargs_with_non_global_operation dictionary or if any of the qubits in qargs
            # are a superset for a non-local operation. For example, if the qargs
            # are (0, 1) and that's a global (ie no non-local operations on (0, 1)
            # operation but there is a non-local operation on (1,) we need to
            # do an extra non-local search for this op to ensure we include any
            # single qubit operation for (1,) as valid. This pattern also holds
            # true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
            # and 1q operations in the same manner)
            if qargs in self._qargs_with_non_global_operation or any(
                frozenset(qargs).issuperset(incomplete_qargs)
                for incomplete_qargs in self._qargs_with_non_global_operation
            ):
                qargs_local_source_basis[frozenset(qargs)].add((node.name, node.op.num_qubits))
            else:
                source_basis.add((node.name, node.op.num_qubits))
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)
                    source_basis, qargs_local_source_basis = self._extract_basis_target(
                        block_dag,
                        {
                            inner: qarg_indices[outer]
                            for inner, outer in zip(block.qubits, node.qargs)
                        },
                        source_basis=source_basis,
                        qargs_local_source_basis=qargs_local_source_basis,
                    )
        return source_basis, qargs_local_source_basis


# this could be singledispatchmethod and included in above class when minimum
# supported python version=3.8.
@singledispatch
def _extract_basis(circuit):
    return circuit


@_extract_basis.register
def _(dag: DAGCircuit):
    for node in dag.op_nodes():
        if not dag.has_calibration_for(node):
            yield (node.name, node.op.num_qubits)
        if isinstance(node.op, ControlFlowOp):
            for block in node.op.blocks:
                yield from _extract_basis(block)


@_extract_basis.register
def _(circ: QuantumCircuit):
    for instr_context in circ.data:
        instr, _, _ = instr_context
        if not circ.has_calibration_for(instr_context):
            yield (instr.name, instr.num_qubits)
        if isinstance(instr, ControlFlowOp):
            for block in instr.blocks:
                yield from _extract_basis(block)


class StopIfBasisRewritable(Exception):
    """Custom exception that signals `retworkx.dijkstra_search` to stop."""


class BasisSearchVisitor(retworkx.visit.DijkstraVisitor):  # pylint: disable=no-member
    """Handles events emitted during `retworkx.dijkstra_search`."""

    def __init__(self, graph, source_basis, target_basis, num_gates_for_rule):
        self.graph = graph
        self.target_basis = set(target_basis)
        self._source_gates_remain = set(source_basis)
        self._num_gates_remain_for_rule = dict(num_gates_for_rule)
        self._basis_transforms = []
        self._predecessors = dict()
        self._opt_cost_map = dict()

    def discover_vertex(self, v, score):
        gate = self.graph[v]
        self._source_gates_remain.discard(gate)
        self._opt_cost_map[gate] = score
        rule = self._predecessors.get(gate, None)
        if rule is not None:
            logger.debug(
                "Gate %s generated using rule \n%s\n with total cost of %s.",
                gate.name,
                rule.circuit,
                score,
            )
            self._basis_transforms.append((gate.name, gate.num_qubits, rule.params, rule.circuit))
        # we can stop the search if we have found all gates in the original ciruit.
        if not self._source_gates_remain:
            # if we start from source gates and apply `basis_transforms` in reverse order, we'll end
            # up with gates in the target basis. Note though that `basis_transforms` may include
            # additional transformations that are not required to map our source gates to the given
            # target basis.
            self._basis_transforms.reverse()
            raise StopIfBasisRewritable

    def examine_edge(self, edge):
        _, target, edata = edge
        if edata is None:
            return

        index = edata["index"]
        self._num_gates_remain_for_rule[index] -= 1

        target = self.graph[target]
        # if there are gates in this `rule` that we have not yet generated, we can't apply
        # this `rule`. if `target` is already in basis, it's not beneficial to use this rule.
        if self._num_gates_remain_for_rule[index] > 0 or target in self.target_basis:
            raise retworkx.visit.PruneSearch  # pylint: disable=no-member

    def edge_relaxed(self, edge):
        _, target, edata = edge
        if edata is not None:
            gate = self.graph[target]
            self._predecessors[gate] = edata["rule"]

    def edge_cost(self, edge):
        """Returns the cost of an edge.

        This function computes the cost of this edge rule by summing
        the costs of all gates in the rule equivalence circuit. In the
        end, we need to subtract the cost of the source since `dijkstra`
        will later add it.
        """

        if edge is None:
            # the target of the edge is a gate in the target basis,
            # so we return a default value of 1.
            return 1

        cost_tot = 0
        rule = edge["rule"]
        for instruction in rule.circuit:
            key = Key(name=instruction.operation.name, num_qubits=len(instruction.qubits))
            cost_tot += self._opt_cost_map[key]

        source = edge["source"]
        return cost_tot - self._opt_cost_map[source]

    @property
    def basis_transforms(self):
        """Returns the gate basis transforms."""
        return self._basis_transforms


def _basis_search(equiv_lib, source_basis, target_basis):
    """Search for a set of transformations from source_basis to target_basis.

    Args:
        equiv_lib (EquivalenceLibrary): Source of valid translations
        source_basis (Set[Tuple[gate_name: str, gate_num_qubits: int]]): Starting basis.
        target_basis (Set[gate_name: str]): Target basis.

    Returns:
        Optional[List[Tuple[gate, equiv_params, equiv_circuit]]]: List of (gate,
            equiv_params, equiv_circuit) tuples tuples which, if applied in order
            will map from source_basis to target_basis. Returns None if no path
            was found.
    """

    logger.debug("Begining basis search from %s to %s.", source_basis, target_basis)

    source_basis = {
        (gate_name, gate_num_qubits)
        for gate_name, gate_num_qubits in source_basis
        if gate_name not in target_basis
    }

    # if source basis is empty, no work to be done.
    if not source_basis:
        return []

    all_gates_in_lib = set()

    graph = retworkx.PyDiGraph()
    nodes_to_indices = dict()
    num_gates_for_rule = dict()

    def lazy_setdefault(key):
        if key not in nodes_to_indices:
            nodes_to_indices[key] = graph.add_node(key)
        return nodes_to_indices[key]

    rcounter = 0  # running sum of the number of equivalence rules in the library.
    for key in equiv_lib._get_all_keys():
        target = lazy_setdefault(key)
        all_gates_in_lib.add(key)
        for equiv in equiv_lib._get_equivalences(key):
            sources = {
                Key(name=instruction.operation.name, num_qubits=len(instruction.qubits))
                for instruction in equiv.circuit
            }
            all_gates_in_lib |= sources
            edges = [
                (
                    lazy_setdefault(source),
                    target,
                    {"index": rcounter, "rule": equiv, "source": source},
                )
                for source in sources
            ]

            num_gates_for_rule[rcounter] = len(sources)
            graph.add_edges_from(edges)
            rcounter += 1

    # This is only neccessary since gates in target basis are currently reported by
    # their names and we need to have in addition the number of qubits they act on.
    target_basis_keys = [
        key
        for gate in target_basis
        for key in filter(lambda key, name=gate: key.name == name, all_gates_in_lib)
    ]

    vis = BasisSearchVisitor(graph, source_basis, target_basis_keys, num_gates_for_rule)
    # we add a dummy node and connect it with gates in the target basis.
    # we'll start the search from this dummy node.
    dummy = graph.add_node("dummy starting node")
    graph.add_edges_from_no_data([(dummy, nodes_to_indices[key]) for key in target_basis_keys])
    rtn = None
    try:
        retworkx.digraph_dijkstra_search(graph, [dummy], vis.edge_cost, vis)
    except StopIfBasisRewritable:
        rtn = vis.basis_transforms

        logger.debug("Transformation path:")
        for gate_name, gate_num_qubits, params, equiv in rtn:
            logger.debug("%s/%s => %s\n%s", gate_name, gate_num_qubits, params, equiv)

    return rtn


def _compose_transforms(basis_transforms, source_basis, source_dag):
    """Compose a set of basis transforms into a set of replacements.

    Args:
        basis_transforms (List[Tuple[gate_name, params, equiv]]): List of
            transforms to compose.
        source_basis (Set[Tuple[gate_name: str, gate_num_qubits: int]]): Names
            of gates which need to be translated.
        source_dag (DAGCircuit): DAG with example gates from source_basis.
            (Used to determine num_params for gate in source_basis.)

    Returns:
        Dict[gate_name, Tuple(params, dag)]: Dictionary mapping between each gate
            in source_basis and a DAGCircuit instance to replace it. Gates in
            source_basis but not affected by basis_transforms will be included
            as a key mapping to itself.
    """
    example_gates = _get_example_gates(source_dag)
    mapped_instrs = {}

    for gate_name, gate_num_qubits in source_basis:
        # Need to grab a gate instance to find num_qubits and num_params.
        # Can be removed following https://github.com/Qiskit/qiskit-terra/pull/3947 .
        example_gate = example_gates[gate_name, gate_num_qubits]
        num_params = len(example_gate.params)

        placeholder_params = ParameterVector(gate_name, num_params)
        placeholder_gate = Gate(gate_name, gate_num_qubits, list(placeholder_params))
        placeholder_gate.params = list(placeholder_params)

        dag = DAGCircuit()
        qr = QuantumRegister(gate_num_qubits)
        dag.add_qreg(qr)
        dag.apply_operation_back(placeholder_gate, qr[:], [])
        mapped_instrs[gate_name, gate_num_qubits] = placeholder_params, dag

    for gate_name, gate_num_qubits, equiv_params, equiv in basis_transforms:
        logger.debug(
            "Composing transform step: %s/%s %s =>\n%s",
            gate_name,
            gate_num_qubits,
            equiv_params,
            equiv,
        )

        for mapped_instr_name, (dag_params, dag) in mapped_instrs.items():
            doomed_nodes = [
                node
                for node in dag.op_nodes()
                if (node.op.name, node.op.num_qubits) == (gate_name, gate_num_qubits)
            ]

            if doomed_nodes and logger.isEnabledFor(logging.DEBUG):

                logger.debug(
                    "Updating transform for mapped instr %s %s from \n%s",
                    mapped_instr_name,
                    dag_params,
                    dag_to_circuit(dag),
                )

            for node in doomed_nodes:

                replacement = equiv.assign_parameters(
                    dict(zip_longest(equiv_params, node.op.params))
                )

                replacement_dag = circuit_to_dag(replacement)

                dag.substitute_node_with_dag(node, replacement_dag)

            if doomed_nodes and logger.isEnabledFor(logging.DEBUG):

                logger.debug(
                    "Updated transform for mapped instr %s %s to\n%s",
                    mapped_instr_name,
                    dag_params,
                    dag_to_circuit(dag),
                )

    return mapped_instrs


def _get_example_gates(source_dag):
    def recurse(dag, example_gates=None):
        example_gates = example_gates or {}
        for node in dag.op_nodes():
            example_gates[(node.op.name, node.op.num_qubits)] = node.op
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    example_gates = recurse(circuit_to_dag(block), example_gates)
        return example_gates

    return recurse(source_dag)
