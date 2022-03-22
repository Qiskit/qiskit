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
            source_basis = set()
            for node in dag.op_nodes():
                if not dag.has_calibration_for(node):
                    source_basis.add((node.name, node.op.num_qubits))
            qargs_local_source_basis = {}
        else:
            basic_instrs = ["barrier", "snapshot"]
            source_basis = set()
            target_basis = self._target.keys() - set(self._non_global_operations)
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
                # and 1q opertaions in the same manner)
                if qargs in self._qargs_with_non_global_operation or any(
                    frozenset(qargs).issuperset(incomplete_qargs)
                    for incomplete_qargs in self._qargs_with_non_global_operation
                ):
                    qargs_local_source_basis[frozenset(qargs)].add((node.name, node.op.num_qubits))
                else:
                    source_basis.add((node.name, node.op.num_qubits))

        target_basis = set(target_basis).union(basic_instrs)

        logger.info(
            "Begin BasisTranslator from source basis %s to target basis %s.",
            source_basis,
            target_basis,
        )

        # Search for a path from source to target basis.
        search_start_time = time.time()
        basis_transforms = _basis_search(
            self._equiv_lib, source_basis, target_basis, _basis_heuristic
        )

        qarg_local_basis_transforms = {}
        for qarg, local_source_basis in qargs_local_source_basis.items():
            expanded_target = target_basis | self._qargs_with_non_global_operation[qarg]
            # For any multiqubit operation that contains a subset of qubits that
            # has a non-local operation, include that non-local operation in the
            # search. This matches with the check we did above to include those
            # subset non-local operations in the check here.
            if len(qarg) > 1:
                for non_local_qarg, local_basis in self._qargs_with_non_global_operation.items():
                    if qarg.issuperset(non_local_qarg):
                        expanded_target |= local_basis

            logger.info(
                "Performing BasisTranslator search from source basis %s to target "
                "basis %s on qarg %s.",
                local_source_basis,
                expanded_target,
                qarg,
            )
            qarg_local_basis_transforms[qarg] = _basis_search(
                self._equiv_lib, local_source_basis, expanded_target, _basis_heuristic
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
        for node in dag.op_nodes():
            node_qargs = tuple(qarg_indices[bit] for bit in node.qargs)
            qubit_set = frozenset(node_qargs)

            if node.name in target_basis:
                continue
            if (
                node_qargs in self._qargs_with_non_global_operation
                and node.name in self._qargs_with_non_global_operation[node_qargs]
            ):
                continue

            if dag.has_calibration_for(node):
                continue

            def replace_node(node, instr_map):
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

            if qubit_set in extra_instr_map:
                replace_node(node, extra_instr_map[qubit_set])
            elif (node.op.name, node.op.num_qubits) in instr_map:
                replace_node(node, instr_map)
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
