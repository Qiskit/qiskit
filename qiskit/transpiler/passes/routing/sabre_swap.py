# -*- coding: utf-8 -*-

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

"""Routing via SWAP insertion using the SABRE method from Li et al."""

import logging
from copy import deepcopy
from itertools import cycle

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode

logger = logging.getLogger(__name__)

W = 0.5  # Weight of extened_set (lookahead window) compared to front_layer.
DELTA = 0.001  # Decay cooefficient for penalizing serial swaps.


class SabreSwap(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The hueristic aims to minimize the number
    of lossy SWAPs inserted and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighoborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, heuristic='basic'):
        """SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'advanced').
        """

        super().__init__()
        self.coupling_map = coupling_map
        self.heuristic = heuristic

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Sabre swap runs on physical circuits only.')

        if len(dag.qubits()) > self.coupling_map.size():
            raise TranspilerError('More virtual qubits exist than physical.')

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = _copy_circuit_metadata(dag)

        # Assume bidirectional couplings, fixing gate direction is easy later.
        self.coupling_map.make_symmetric()

        canonical_register = dag.qregs['q']
        current_layout = Layout.generate_trivial_layout(canonical_register)

        # Set the max number of gates in the lookahead window to dag width.
        EXTENDED_SET_SIZE = 4  # len(current_layout)

        # Start algorithm from the front layer and iterate until all gates done.
        front_layer = dag.front_layer()
        applied_gates = set()
        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    physical_qubits = (current_layout[v0], current_layout[v1])
                    if physical_qubits in self.coupling_map.get_edges():
                        execute_gate_list.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)

            if execute_gate_list:
                for node in execute_gate_list:
                    new_node = _transform_gate_for_layout(node, current_layout)
                    mapped_dag.apply_operation_back(new_node.op,
                                                    new_node.qargs,
                                                    new_node.cargs,
                                                    new_node.condition)
                    front_layer.remove(node)
                    applied_gates.add(node)
                    for successor in dag.quantum_successors(node):
                        if successor.type != 'op':
                            continue
                        elif _is_resolved(successor, dag, applied_gates):
                            front_layer.append(successor)

                # Diagnostics
                logger.debug('free! %s',
                             [(n.name, n.qargs) for n in execute_gate_list])
                logger.debug('front_layer: %s',
                             [(n.name, n.qargs) for n in front_layer])

                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it.
            else:
                extended_set = _obtain_extended_set(dag, front_layer,
                                                    EXTENDED_SET_SIZE)

                swap_candidate_list = _obtain_swaps(front_layer, current_layout,
                                                    self.coupling_map)

                swap_scores = []
                for i, swap in enumerate(swap_candidate_list):
                    trial_layout = current_layout.copy()
                    trial_layout.swap(*swap)
                    score = _score_heuristic(front_layer,
                                             extended_set,
                                             trial_layout,
                                             self.coupling_map,
                                             self.heuristic)
                    swap_scores.append(score)
                min_score = min(swap_scores)
                best_swap = swap_candidate_list[swap_scores.index(min_score)]
                swap_node = DAGNode(op=SwapGate(), qargs=best_swap, type='op')
                swap_node = _transform_gate_for_layout(swap_node, current_layout)
                mapped_dag.apply_operation_back(swap_node.op, swap_node.qargs)
                current_layout.swap(*best_swap)

                # Diagnostics
                logger.debug('SWAP Selection...')
                logger.debug('extended_set: %s',
                             [(n.name, n.qargs) for n in extended_set])
                logger.debug('swap scores: %s',
                             [(swap_candidate_list[i], swap_scores[i])
                              for i in range(len(swap_scores))])
                logger.debug('best swap: %s', best_swap)

        self.property_set['final_layout'] = current_layout

        return mapped_dag


def _is_resolved(node, dag, applied_gates):
    """Return True if all of a node's predecessors in dag are applied.
    """
    predecessors = dag.quantum_predecessors(node)
    predecessors = filter(lambda x: x.type=='op', predecessors)
    if all([n in applied_gates for n in predecessors]):
        return True
    else:
        return False


def _obtain_extended_set(dag, front_layer, window_size):
    """Populate extended_set by looking ahead a fixed number of gates.
    For each existing element add a successor until reaching limit.
    """
    # TODO: use layers instead of bfs_successors so long range successors aren't included.
    extended_set = set()
    bfs_successors_pernode = [dag.bfs_successors(n) for n in front_layer]
    node_lookahead_exhausted = [False] * len(front_layer)
    for i, node_successor_generator in cycle(enumerate(bfs_successors_pernode)):
        if all(node_lookahead_exhausted) or len(extended_set) >= window_size:
            break

        try:
            _, successors = next(node_successor_generator)
            successors = list(filter(lambda x: x.type=='op' and len(x.qargs)==2,
                                     successors))
        except StopIteration:
            node_lookahead_exhausted[i] = True
            continue

        successors = iter(successors)
        while len(extended_set) < window_size:
            try:
                extended_set.add(next(successors))
            except StopIteration:
                break

    return extended_set


def _obtain_swaps(front_layer, current_layout, coupling_map):
    """Return a list of candidate swaps that affect qubits in front_layer.

    For each virtual qubit in front_layer, find its current location
    on hardware and the physical qubits in that neighborhood. Every SWAP
    on virtual qubits that corresponds to one of those physical couplings
    is a candidate SWAP.
    """
    candidate_swaps = set()
    for node in front_layer:
        for virtual in node.qargs:
            physical = current_layout[virtual]
            for neighbor in coupling_map.neighbors(physical):
                virtual_neighbor = current_layout[neighbor]
                candidate_swaps.add((virtual, virtual_neighbor))  # TODO: sort so i,j is seen once.
            
    return list(candidate_swaps)


def _score_heuristic(front_layer, extended_set, layout, coupling_map, heuristic):
    """Return a heuristic score for a trial layout.

    Assuming a trial layout has resulted from a SWAP, we now assign a cost
    to it. The goodness of a layout is evaluated based on how viable it makes
    the remaining virtual gates that must be applied.

    Basic cost function:
    The sum of distances for corresponding physical qubits of
    interacting virtual qubits in the front_layer.

    .. math::
        
        H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]

    Advanced cost function:
    This is the sum of two costs: first is the same as the basic cost.
    Second is the basic cost but now evaluated for the
    extended set as well (i.e. |E| number of upcoming successors to gates in
    front_layer F). This is weighted by some amount W to signify that
    upcoming gates are less important that the front_layer.
    The whole cost is multiplied by a decay factor. This increases the cost
    if the SWAP that generated this trial layout was recently used (i.e.
    it penalizes increase in depth).
        
    .. math::
        
        H_{advanced} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
            \frac{1}{\abs{F}} \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
            + W * \frac{1}{\abs{E}} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]
            }
    """
    # TODO: add decay
    if heuristic == 'basic':
        return sum(coupling_map.distance(*[layout[q] for q in node.qargs])
                   for node in front_layer)
    elif heuristic == 'advanced':
        W = 0.5

        first_cost = _score_heuristic(front_layer, [], layout, coupling_map, 'basic')
        first_cost /= len(front_layer)

        second_cost = _score_heuristic(extended_set, [], layout, coupling_map, 'basic')
        second_cost = 0.0 if not extended_set else second_cost / len(extended_set)

        return first_cost + W * second_cost
    else:
        raise TranspilerError('Heuristic %s not recognized.' % heuristic)        


def _copy_circuit_metadata(source_dag):
    """Return a copy of source_dag with metadata but empty.
    """
    target_dag = DAGCircuit()
    target_dag.name = source_dag.name

    for qreg in source_dag.qregs.values():
        target_dag.add_qreg(qreg)
    for creg in source_dag.cregs.values():
        target_dag.add_creg(creg)

    return target_dag


def _transform_gate_for_layout(op_node, layout):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = deepcopy(op_node)

    device_qreg = op_node.qargs[0].register
    premap_qargs = op_node.qargs
    mapped_qargs = map(lambda x: device_qreg[layout[x]], premap_qargs)
    mapped_op_node.qargs = mapped_op_node.op.qargs = list(mapped_qargs)

    return mapped_op_node
