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

"""Map input circuit onto a backend topology via insertion of SWAPs."""

import logging
from copy import deepcopy

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGNode

logger = logging.getLogger(__name__)


class LookaheadSwap(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of Sven Jandura's swap mapper submission for the 2018 Qiskit
    Developer Challenge, adapted to integrate into the transpiler architecture.

    The role of the swapper pass is to modify the starting circuit to be compatible
    with the target device's topology (the set of two-qubit gates available on the
    hardware.) To do this, the pass will insert SWAP gates to relocate the virtual
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
    - For the four (search_width) highest-ranking SWAPs, repeat the above process on
      the layout that would be generated if they were applied.
    - Repeat this process down to a depth of four (search_depth) SWAPs away from the
      initial layout, for a total of 256 (search_width^search_depth) prospective
      layouts.
    - Choose the layout which maximizes the number of two-qubit which could be
      performed. Add its mapped gates, including the SWAPs generated, to the
      output circuit.
    - Repeat the above until all gates from the initial circuit are mapped.

    For more details on the algorithm, see Sven's blog post:
    https://medium.com/qiskit/improving-a-quantum-compiler-48410d7a7084
    """

    def __init__(self, coupling_map, search_depth=4, search_width=4, fake_run=False):
        """LookaheadSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            search_depth (int): lookahead tree depth when ranking best SWAP options.
            search_width (int): lookahead tree width when ranking best SWAP options.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
        """

        super().__init__()
        self.coupling_map = coupling_map
        self.search_depth = search_depth
        self.search_width = search_width
        self.fake_run = fake_run

    def run(self, dag):
        """Run the LookaheadSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map in
                the property_set.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Lookahead swap runs on physical circuits only")

        number_of_available_qubits = len(self.coupling_map.physical_qubits)
        if len(dag.qubits) > number_of_available_qubits:
            raise TranspilerError(
                f"The number of DAG qubits ({len(dag.qubits)}) is greater than the number of "
                f"available device qubits ({number_of_available_qubits})."
            )

        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        mapped_gates = []
        ordered_virtual_gates = list(dag.serial_layers())
        gates_remaining = ordered_virtual_gates.copy()

        while gates_remaining:
            logger.debug("Top-level routing step: %d gates remaining.", len(gates_remaining))

            best_step = _search_forward_n_swaps(
                current_layout,
                gates_remaining,
                self.coupling_map,
                self.search_depth,
                self.search_width,
            )

            if best_step is None:
                raise TranspilerError(
                    "Lookahead failed to find a swap which mapped "
                    "gates or improved layout score."
                )

            logger.debug(
                "Found best step: mapped %d gates. Added swaps: %s.",
                len(best_step["gates_mapped"]),
                best_step["swaps_added"],
            )

            current_layout = best_step["layout"]
            gates_mapped = best_step["gates_mapped"]
            gates_remaining = best_step["gates_remaining"]

            mapped_gates.extend(gates_mapped)

        if self.fake_run:
            self.property_set["final_layout"] = current_layout
            return dag

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = dag._copy_circuit_metadata()

        for node in mapped_gates:
            mapped_dag.apply_operation_back(op=node.op, qargs=node.qargs, cargs=node.cargs)

        return mapped_dag


def _search_forward_n_swaps(layout, gates, coupling_map, depth, width):
    """Search for SWAPs which allow for application of largest number of gates.

    Args:
        layout (Layout): Map from virtual qubit index to physical qubit index.
        gates (list): Gates to be mapped.
        coupling_map (CouplingMap): CouplingMap of the target backend.
        depth (int): Number of SWAP layers to search before choosing a result.
        width (int): Number of SWAPs to consider at each layer.
    Returns:
        optional(dict): Describes solution step found. If None, no swaps leading
            to an improvement were found. Keys:
            layout (Layout): Virtual to physical qubit map after SWAPs.
            swaps_added (list): List of qargs of swap gates introduced.
            gates_remaining (list): Gates that could not be mapped.
            gates_mapped (list): Gates that were mapped, including added SWAPs.

    """
    gates_mapped, gates_remaining = _map_free_gates(layout, gates, coupling_map)

    base_step = {
        "layout": layout,
        "swaps_added": [],
        "gates_mapped": gates_mapped,
        "gates_remaining": gates_remaining,
    }

    if not gates_remaining or depth == 0:
        return base_step

    # Include symmetric 2q gates (e.g coupling maps with both [0,1] and [1,0])
    # as one available swap.
    possible_swaps = {tuple(sorted(edge)) for edge in coupling_map.get_edges()}

    def _score_swap(swap):
        """Calculate the relative score for a given SWAP."""
        trial_layout = layout.copy()
        trial_layout.swap(*swap)
        return _calc_layout_distance(gates, coupling_map, trial_layout)

    ranked_swaps = sorted(possible_swaps, key=_score_swap)
    logger.debug(
        "At depth %d, ranked candidate swaps: %s...",
        depth,
        [(swap, _score_swap(swap)) for swap in ranked_swaps[: width * 2]],
    )

    best_swap, best_step = None, None
    for rank, swap in enumerate(ranked_swaps):
        trial_layout = layout.copy()
        trial_layout.swap(*swap)
        next_step = _search_forward_n_swaps(
            trial_layout, gates_remaining, coupling_map, depth - 1, width
        )

        if next_step is None:
            continue

        # ranked_swaps already sorted by distance, so distance is the tie-breaker.
        if best_swap is None or _score_step(next_step) > _score_step(best_step):
            logger.debug(
                "At depth %d, updating best step: %s (score: %f).",
                depth,
                [swap] + next_step["swaps_added"],
                _score_step(next_step),
            )
            best_swap, best_step = swap, next_step

        if (
            rank >= min(width, len(ranked_swaps) - 1)
            and best_step is not None
            and (
                len(best_step["gates_mapped"]) > depth
                or len(best_step["gates_remaining"]) < len(gates_remaining)
                or (
                    _calc_layout_distance(
                        best_step["gates_remaining"], coupling_map, best_step["layout"]
                    )
                    < _calc_layout_distance(gates_remaining, coupling_map, layout)
                )
            )
        ):
            # Once we've examined either $WIDTH swaps, or all available swaps,
            # return the best-scoring swap provided it leads to an improvement
            # in either the number of gates mapped, number of gates left to be
            # mapped, or in the score of the ending layout.
            break
    else:
        return None

    logger.debug("At depth %d, best_swap set: %s.", depth, [best_swap] + best_step["swaps_added"])

    best_swap_gate = _swap_ops_from_edge(best_swap, layout)
    return {
        "layout": best_step["layout"],
        "swaps_added": [best_swap] + best_step["swaps_added"],
        "gates_remaining": best_step["gates_remaining"],
        "gates_mapped": gates_mapped + best_swap_gate + best_step["gates_mapped"],
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
        if not gate["partition"]:
            qubits = [n for n in gate["graph"].nodes() if n.type == "op"][0].qargs

            if not qubits:
                continue

            if blocked_qubits.intersection(qubits):
                blocked_qubits.update(qubits)
                remaining_gates.append(gate)
            else:
                mapped_gate = _transform_gate_for_layout(gate, layout)
                mapped_gates.append(mapped_gate)
            continue

        qubits = gate["partition"][0]

        if blocked_qubits.intersection(qubits):
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)
        elif len(qubits) == 1:
            mapped_gate = _transform_gate_for_layout(gate, layout)
            mapped_gates.append(mapped_gate)
        elif coupling_map.distance(*(layout[q] for q in qubits)) == 1:
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

    return sum(
        coupling_map.distance(*(layout[q] for q in gate["partition"][0]))
        for gate in gates[:max_gates]
        if gate["partition"] and len(gate["partition"][0]) == 2
    )


def _score_step(step):
    """Count the mapped two-qubit gates, less the number of added SWAPs."""
    # Each added swap will add 3 ops to gates_mapped, so subtract 3.
    return len([g for g in step["gates_mapped"] if len(g.qargs) == 2]) - 3 * len(
        step["swaps_added"]
    )


def _transform_gate_for_layout(gate, layout):
    """Return op implementing a virtual gate on given layout."""
    mapped_op_node = deepcopy([n for n in gate["graph"].nodes() if n.type == "op"][0])

    device_qreg = QuantumRegister(len(layout.get_physical_bits()), "q")
    mapped_qargs = [device_qreg[layout[a]] for a in mapped_op_node.qargs]
    mapped_op_node.qargs = mapped_qargs

    return mapped_op_node


def _swap_ops_from_edge(edge, layout):
    """Generate list of ops to implement a SWAP gate along a coupling edge."""
    device_qreg = QuantumRegister(len(layout.get_physical_bits()), "q")
    qreg_edge = [device_qreg[i] for i in edge]

    # TODO shouldn't be making other nodes not by the DAG!!
    return [DAGNode(op=SwapGate(), qargs=qreg_edge, cargs=[], type="op")]
