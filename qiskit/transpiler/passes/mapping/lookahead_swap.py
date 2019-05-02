# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Implementation of Sven Jandura's swap mapper submission for the 2018 QISKit
Developer Challenge, adapted to integrate into the transpiler architecture.

The role of the mapper pass is to modify the starting circuit to be compatible
with the target device's topology (the set of two-qubit gates available on the
hardware.) To do this, the mapper will insert SWAP gates to relocate the virtual
qubits for each upcoming gate onto a set of coupled physical qubits. However, as
SWAP gates are particularly lossy, the goal is to accomplish this remapping while
introducing the fewest possible additional SWAPs.

This algorithm searches through the available combinations of SWAP gates by means
of a narrowed best first/beam search, described as follows:

- Start with a layout of virtual qubits onto physical qubits.
- Find any gates in the input circuit which can be performed with the current
  layout and mark them as mapped.
- For all possible SWAP gates, calculate the layout that would result from their
  application and rank them according to the distance of the resulting layout
  over upcoming gates (see _calc_layout_distance.)
- For the four (SEARCH_WIDTH) highest-ranking SWAPs, repeat the above process on
  the layout that would be generated if they were applied.
- Repeat this process down to a depth of four (SEARCH_DEPTH) SWAPs away from the
  initial layout, for a total of 256 (SEARCH_WIDTH^SEARCH_DEPTH) prospective
  layouts.
- Choose the layout which maximizes the number of two-qubit which could be
  performed. Add its mapped gates, including the SWAPs generated, to the
  output circuit.
- Repeat the above until all gates from the initial circuit are mapped.

For more details on the algorithm, see Sven's blog post:
https://medium.com/qiskit/improving-a-quantum-compiler-48410d7a7084

"""

from copy import deepcopy

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler import Layout
from qiskit.dagcircuit import DAGNode

from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements

SEARCH_DEPTH = 4
SEARCH_WIDTH = 4


class LookaheadSwap(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs."""

    def __init__(self, coupling_map, initial_layout=None):
        """Initialize a LookaheadSwap instance.

        Arguments:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            initial_layout (Layout): The initial layout of the DAG to analyze.
        """

        super().__init__()
        self._coupling_map = coupling_map
        self.initial_layout = initial_layout

    def run(self, dag):
        """Run one pass of the lookahead mapper on the provided DAG.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map in
                the property_set.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        coupling_map = self._coupling_map
        ordered_virtual_gates = list(dag.serial_layers())

        if self.initial_layout is None:
            if self.property_set["layout"]:
                self.initial_layout = self.property_set["layout"]
            else:
                self.initial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        if len(dag.qubits()) != len(self.initial_layout):
            raise TranspilerError('The layout does not match the amount of qubits in the DAG')

        if len(self._coupling_map.physical_qubits) != len(self.initial_layout):
            raise TranspilerError(
                "Mappers require to have the layout to be the same size as the coupling map")

        mapped_gates = []
        layout = self.initial_layout.copy()
        gates_remaining = ordered_virtual_gates.copy()

        while gates_remaining:
            best_step = _search_forward_n_swaps(layout, gates_remaining,
                                                coupling_map)

            layout = best_step['layout']
            gates_mapped = best_step['gates_mapped']
            gates_remaining = best_step['gates_remaining']

            mapped_gates.extend(gates_mapped)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = _copy_circuit_metadata(dag, coupling_map)

        for node in mapped_gates:
            mapped_dag.apply_operation_back(op=node.op, qargs=node.qargs, cargs=node.cargs)

        return mapped_dag


def _search_forward_n_swaps(layout, gates, coupling_map,
                            depth=SEARCH_DEPTH, width=SEARCH_WIDTH):
    """Search for SWAPs which allow for application of largest number of gates.

    Arguments:
        layout (Layout): Map from virtual qubit index to physical qubit index.
        gates (list): Gates to be mapped.
        coupling_map (CouplingMap): CouplingMap of the target backend.
        depth (int): Number of SWAP layers to search before choosing a result.
        width (int): Number of SWAPs to consider at each layer.
    Returns:
        dict: Describes solution step found.
            layout (Layout): Virtual to physical qubit map after SWAPs.
            gates_remaining (list): Gates that could not be mapped.
            gates_mapped (list): Gates that were mapped, including added SWAPs.

    """

    gates_mapped, gates_remaining = _map_free_gates(layout, gates, coupling_map)

    base_step = {'layout': layout,
                 'swaps_added': 0,
                 'gates_mapped': gates_mapped,
                 'gates_remaining': gates_remaining}

    if not gates_remaining or depth == 0:
        return base_step

    possible_swaps = coupling_map.get_edges()

    def _score_swap(swap):
        """Calculate the relative score for a given SWAP."""
        trial_layout = layout.copy()
        trial_layout.swap(*swap)
        return _calc_layout_distance(gates, coupling_map, trial_layout)

    ranked_swaps = sorted(possible_swaps, key=_score_swap)

    best_swap, best_step = None, None
    for swap in ranked_swaps[:width]:
        trial_layout = layout.copy()
        trial_layout.swap(*swap)
        next_step = _search_forward_n_swaps(trial_layout, gates_remaining,
                                            coupling_map, depth - 1, width)

        # ranked_swaps already sorted by distance, so distance is the tie-breaker.
        if best_swap is None or _score_step(next_step) > _score_step(best_step):
            best_swap, best_step = swap, next_step

    best_swap_gate = _swap_ops_from_edge(best_swap, layout)
    return {
        'layout': best_step['layout'],
        'swaps_added': 1 + best_step['swaps_added'],
        'gates_remaining': best_step['gates_remaining'],
        'gates_mapped': gates_mapped + best_swap_gate + best_step['gates_mapped'],
    }


def _map_free_gates(layout, gates, coupling_map):
    """Map all gates that can be executed with the current layout.

    Args:
        layout (Layout): Map from virtual qubit index to physical qubit index.
        gates (list): Gates to be mapped.
        coupling_map (CouplingMap): CouplingMap for target device topology.

    Returns:
        tuple:
            mapped_gates (list): ops for gates that can be executed, mapped onto layout.
            remaining_gates (list): gates that cannot be executed on the layout.

    """

    blocked_qubits = set()

    mapped_gates = []
    remaining_gates = []

    for gate in gates:
        # Gates without a partition (barrier, snapshot, save, load, noise) may
        # still have associated qubits. Look for them in the qargs.
        if not gate['partition']:
            qubits = [n for n in gate['graph'].nodes() if n.type == 'op'][0].qargs

            if not qubits:
                continue

            if blocked_qubits.intersection(qubits):
                blocked_qubits.update(qubits)
                remaining_gates.append(gate)
            else:
                mapped_gate = _transform_gate_for_layout(gate, layout)
                mapped_gates.append(mapped_gate)
            continue

        qubits = gate['partition'][0]

        if blocked_qubits.intersection(qubits):
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)
        elif len(qubits) == 1:
            mapped_gate = _transform_gate_for_layout(gate, layout)
            mapped_gates.append(mapped_gate)
        elif coupling_map.distance(*[layout[q] for q in qubits]) == 1:
            mapped_gate = _transform_gate_for_layout(gate, layout)
            mapped_gates.append(mapped_gate)
        else:
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)

    return mapped_gates, remaining_gates


def _calc_layout_distance(gates, coupling_map, layout, max_gates=None):
    """Return the sum of the distances of two-qubit pairs in each CNOT in gates
    according to the layout and the coupling.
    """

    if max_gates is None:
        max_gates = 50 + 10 * len(coupling_map.physical_qubits)

    return sum(coupling_map.distance(*[layout[q] for q in gate['partition'][0]])
               for gate in gates[:max_gates]
               if gate['partition'] and len(gate['partition'][0]) == 2)


def _score_step(step):

    """Count the mapped two-qubit gates, less the number of added SWAPs."""
    # Each added swap will add 3 ops to gates_mapped, so subtract 3.
    return len([g for g in step['gates_mapped']
                if len(g.qargs) == 2]) - 3 * step['swaps_added']


def _copy_circuit_metadata(source_dag, coupling_map):
    """Return a copy of source_dag with metadata but empty.
    Generate only a single qreg in the output DAG, matching the size of the
    coupling_map."""

    target_dag = DAGCircuit()
    target_dag.name = source_dag.name

    for creg in source_dag.cregs.values():
        target_dag.add_creg(creg)

    device_qreg = QuantumRegister(len(coupling_map.physical_qubits), 'q')
    target_dag.add_qreg(device_qreg)

    return target_dag


def _transform_gate_for_layout(gate, layout):
    """Return op implementing a virtual gate on given layout."""

    mapped_op_node = deepcopy([n for n in gate['graph'].nodes() if n.type == 'op'][0])

    # Workaround until #1816, apply mapped to qargs to both DAGNode and op
    device_qreg = QuantumRegister(len(layout.get_physical_bits()), 'q')
    mapped_qargs = [(device_qreg, layout[a]) for a in mapped_op_node.qargs]
    mapped_op_node.qargs = mapped_op_node.op.qargs = mapped_qargs

    mapped_op_node.pop('name')

    return mapped_op_node


def _swap_ops_from_edge(edge, layout):
    """Generate list of ops to implement a SWAP gate along a coupling edge."""

    device_qreg = QuantumRegister(len(layout.get_physical_bits()), 'q')
    qreg_edge = [(device_qreg, i) for i in edge]

    # TODO shouldn't be making other nodes not by the DAG!!
    return [
        DAGNode({'op': SwapGate(), 'qargs': qreg_edge, 'cargs': [], 'type': 'op'})
    ]
