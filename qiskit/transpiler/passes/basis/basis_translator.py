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

"""Translates gates to a target basis using a given equivalence library."""

import time
import logging

from itertools import zip_longest

import retworkx

from qiskit.circuit import Gate, ParameterVector, QuantumRegister
from qiskit.circuit.equivalence import Key
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


logger = logging.getLogger(__name__)


class BasisTranslator(TransformationPass):
    """Translates gates to a target basis by searching for a set of translations
    from a given EquivalenceLibrary.

    This pass operates in several steps:

    * Determine the source basis from the input circuit.
    * Perform an A* search over basis sets, starting from the source basis and
      targeting the device's target_basis, with edges discovered from the
      provided EquivalenceLibrary. The heuristic used by the A* search is the
      number of distinct circuit basis gates not in the target_basis, plus the
      number of distinct device basis gates not used in the current basis.
    * The found path, as a set of rules from the EquivalenceLibrary, is composed
      into a set of gate replacement rules.
    * The composed replacement rules are applied in-place to each op node which
      is not already in the target_basis.

    """

    def __init__(self, equivalence_library, target_basis):
        """Initialize a BasisTranslator instance.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            target_basis (list[str]): Target basis names to unroll to, e.g. `['u3', 'cx']`.
        """

        super().__init__()

        self._equiv_lib = equivalence_library
        self._target_basis = target_basis

    def run(self, dag):
        """Translate an input DAGCircuit to the target basis.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            TranspilerError: if the target basis cannot be reached

        Returns:
            DAGCircuit: translated circuit.
        """

        if self._target_basis is None:
            return dag

        # Names of instructions assumed to supported by any backend.
        basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay"]

        target_basis = set(self._target_basis).union(basic_instrs)

        source_basis = set()
        for node in dag.op_nodes():
            if not dag.has_calibration_for(node) and not node.name in target_basis:
                source_basis.add((node.name, node.op.num_qubits))

        logger.info(
            "Begin BasisTranslator from source basis %s to target basis %s.",
            source_basis,
            target_basis,
        )

        # Search for a path from source to target basis.

        search_start_time = time.time()
        basis_transforms = _basis_search(self._equiv_lib, source_basis, target_basis)
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

        compose_end_time = time.time()
        logger.info(
            "Basis translation paths composed in %.3fs.", compose_end_time - compose_start_time
        )

        # Replace source instructions with target translations.

        replace_start_time = time.time()
        for node in dag.op_nodes():
            if node.name in target_basis:
                continue

            if dag.has_calibration_for(node):
                continue

            if (node.op.name, node.op.num_qubits) in instr_map:
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
                    from qiskit.converters import dag_to_circuit, circuit_to_dag

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
                    if node.op.condition:
                        dag_op = dag_op.copy()
                    dag.substitute_node(node, dag_op, inplace=True)

                    if bound_target_dag.global_phase:
                        dag.global_phase += bound_target_dag.global_phase
                else:
                    dag.substitute_node_with_dag(node, bound_target_dag)
            else:
                raise TranspilerError(f"BasisTranslator did not map {node.name}.")

        replace_end_time = time.time()
        logger.info(
            "Basis translation instructions replaced in %.3fs.",
            replace_end_time - replace_start_time,
        )

        return dag


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

    # if source basis is empty, no work to be done.
    if not source_basis:
        return []

    class StopIfBasisRewritable(Exception):
        pass

    class BasisSearchVisitor(retworkx.visit.DijkstraVisitor):
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
                self._basis_transforms.append(
                    (gate.name, gate.num_qubits, rule.params, rule.circuit)
                )
            # we can stop the search if we have found all gates in the original ciruit.
            if not self._source_gates_remain:
                # if we start from source gates and apply `basis_transforms` in reverse order, we'll end up with
                # gates in the target basis. Note though that `basis_transforms` may include additional transformations
                # that are not required to map our source gates to the given target basis.
                self._basis_transforms.reverse()
                raise StopIfBasisRewritable

        def examine_edge(self, edge):
            _, target, edata = edge
            if edata is None:
                return

            index = edata["index"]
            self._num_gates_remain_for_rule[index] -= 1

            target = self.graph[target]
            # if there are gates in this `rule` that we have not yet generated, we can't apply this `rule`.
            # if `target` is already in basis, it's not beneficial to use this rule.
            if self._num_gates_remain_for_rule[index] > 0 or target in self.target_basis:
                raise retworkx.visit.PruneSearch

        def edge_relaxed(self, edge):
            _, target, edata = edge
            if edata is not None:
                gate = self.graph[target]
                self._predecessors[gate] = edata["rule"]

        # This function computes the cost of this edge rule by summing
        # the costs of all gates in the rule equivalence circuit. In the
        # end, we need to subtract the cost of the source since `dijkstra`
        # will later add it.
        def edge_cost(self, edge):
            if edge is None:
                # the target of the edge is a gate in the target basis,
                # so we return a default value of 1.
                return 1

            cost_tot = 0
            rule = edge["rule"]
            for gate, qargs, _ in rule.circuit:
                key = Key(name=gate.name, num_qubits=len(qargs))
                cost_tot += self._opt_cost_map[key]

            source = edge["source"]
            return cost_tot - self._opt_cost_map[source]

        @property
        def basis_transforms(self):
            return self._basis_transforms

    all_gates_in_lib = set()

    graph = retworkx.PyDiGraph()
    nodes_to_indices = dict()
    num_gates_for_rule = dict()

    def get_nid_or_insert(key):
        if key not in nodes_to_indices:
            nodes_to_indices[key] = graph.add_node(key)
        return nodes_to_indices[key]

    rcounter = 0  # running sum of the number of equivalence rules in the library.
    for key in equiv_lib._get_all_keys():
        target = get_nid_or_insert(key)
        all_gates_in_lib.add(key)
        for equiv in equiv_lib._get_equivalences(key):
            sources = set(
                [Key(name=gate.name, num_qubits=len(qargs)) for gate, qargs, _ in equiv.circuit]
            )
            all_gates_in_lib |= sources
            edges = [
                (
                    get_nid_or_insert(source),
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
        for key in filter(lambda key: key.name == gate, all_gates_in_lib)
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

    example_gates = {(node.op.name, node.op.num_qubits): node.op for node in source_dag.op_nodes()}
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
                from qiskit.converters import dag_to_circuit

                logger.debug(
                    "Updating transform for mapped instr %s %s from \n%s",
                    mapped_instr_name,
                    dag_params,
                    dag_to_circuit(dag),
                )

            for node in doomed_nodes:
                from qiskit.converters import circuit_to_dag

                replacement = equiv.assign_parameters(
                    dict(zip_longest(equiv_params, node.op.params))
                )

                replacement_dag = circuit_to_dag(replacement)

                dag.substitute_node_with_dag(node, replacement_dag)

            if doomed_nodes and logger.isEnabledFor(logging.DEBUG):
                from qiskit.converters import dag_to_circuit

                logger.debug(
                    "Updated transform for mapped instr %s %s to\n%s",
                    mapped_instr_name,
                    dag_params,
                    dag_to_circuit(dag),
                )

    return mapped_instrs
