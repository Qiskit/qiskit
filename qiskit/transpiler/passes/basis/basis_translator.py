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

from heapq import heappush, heappop
from itertools import zip_longest
from itertools import count as iter_count
from collections import defaultdict

import numpy as np

from qiskit.circuit import Gate, ParameterVector, QuantumRegister
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
            if not dag.has_calibration_for(node):
                source_basis.add((node.name, node.op.num_qubits))

        logger.info(
            "Begin BasisTranslator from source basis %s to target " "basis %s.",
            source_basis,
            target_basis,
        )

        # Search for a path from source to target basis.

        search_start_time = time.time()
        basis_transforms = _basis_search(
            self._equiv_lib, source_basis, target_basis, _basis_heuristic
        )
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


def _basis_heuristic(basis, target):
    """Simple metric to gauge distance between two bases as the number of
    elements in the symmetric difference of the circuit basis and the device
    basis.
    """
    return len({gate_name for gate_name, gate_num_qubits in basis} ^ target)


def _basis_search(equiv_lib, source_basis, target_basis, heuristic):
    """Search for a set of transformations from source_basis to target_basis.

    Args:
        equiv_lib (EquivalenceLibrary): Source of valid translations
        source_basis (Set[Tuple[gate_name: str, gate_num_qubits: int]]): Starting basis.
        target_basis (Set[gate_name: str]): Target basis.
        heuristic (Callable[[source_basis, target_basis], int]): distance heuristic.

    Returns:
        Optional[List[Tuple[gate, equiv_params, equiv_circuit]]]: List of (gate,
            equiv_params, equiv_circuit) tuples tuples which, if applied in order
            will map from source_basis to target_basis. Returns None if no path
            was found.
    """

    source_basis = frozenset(source_basis)
    target_basis = frozenset(target_basis)

    open_set = set()  # Bases found but not yet inspected.
    closed_set = set()  # Bases found and inspected.

    # Priority queue for inspection order of open_set. Contains Tuple[priority, count, basis]
    open_heap = []

    # Map from bases in closed_set to predecessor with lowest cost_from_source.
    # Values are Tuple[prev_basis, gate_name, params, circuit].
    came_from = {}

    basis_count = iter_count()  # Used to break ties in priority.

    open_set.add(source_basis)
    heappush(open_heap, (0, next(basis_count), source_basis))

    # Map from basis to lowest found cost from source.
    cost_from_source = defaultdict(lambda: np.inf)
    cost_from_source[source_basis] = 0

    # Map from basis to cost_from_source + heuristic.
    est_total_cost = defaultdict(lambda: np.inf)
    est_total_cost[source_basis] = heuristic(source_basis, target_basis)

    logger.debug("Begining basis search from %s to %s.", source_basis, target_basis)

    while open_set:
        _, _, current_basis = heappop(open_heap)

        if current_basis in closed_set:
            # When we close a node, we don't remove it from the heap,
            # so skip here.
            continue

        if {gate_name for gate_name, gate_num_qubits in current_basis}.issubset(target_basis):
            # Found target basis. Construct transform path.
            rtn = []
            last_basis = current_basis
            while last_basis != source_basis:
                prev_basis, gate_name, gate_num_qubits, params, equiv = came_from[last_basis]

                rtn.append((gate_name, gate_num_qubits, params, equiv))
                last_basis = prev_basis
            rtn.reverse()

            logger.debug("Transformation path:")
            for gate_name, gate_num_qubits, params, equiv in rtn:
                logger.debug("%s/%s => %s\n%s", gate_name, gate_num_qubits, params, equiv)
            return rtn

        logger.debug("Inspecting basis %s.", current_basis)
        open_set.remove(current_basis)
        closed_set.add(current_basis)

        for gate_name, gate_num_qubits in current_basis:
            if gate_name in target_basis:
                continue

            equivs = equiv_lib._get_equivalences((gate_name, gate_num_qubits))

            basis_remain = current_basis - {(gate_name, gate_num_qubits)}
            neighbors = [
                (
                    frozenset(
                        basis_remain
                        | {(inst.name, inst.num_qubits) for inst, qargs, cargs in equiv.data}
                    ),
                    params,
                    equiv,
                )
                for params, equiv in equivs
            ]

            # Weight total path length of transformation weakly.
            tentative_cost_from_source = cost_from_source[current_basis] + 1e-3

            for neighbor, params, equiv in neighbors:
                if neighbor in closed_set:
                    continue

                if tentative_cost_from_source >= cost_from_source[neighbor]:
                    continue

                open_set.add(neighbor)
                came_from[neighbor] = (current_basis, gate_name, gate_num_qubits, params, equiv)
                cost_from_source[neighbor] = tentative_cost_from_source
                est_total_cost[neighbor] = tentative_cost_from_source + heuristic(
                    neighbor, target_basis
                )
                heappush(open_heap, (est_total_cost[neighbor], next(basis_count), neighbor))

    return None


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
