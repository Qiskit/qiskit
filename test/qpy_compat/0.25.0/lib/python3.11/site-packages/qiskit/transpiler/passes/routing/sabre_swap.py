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
from copy import copy, deepcopy

import rustworkx

from qiskit.circuit import SwitchCaseOp, ControlFlowOp, Clbit, ClassicalRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.tools.parallel import CPU_COUNT

from qiskit._accelerate.sabre_swap import (
    build_swap_map,
    Heuristic,
    NeighborTable,
    SabreDAG,
)
from qiskit._accelerate.nlayout import NLayout

logger = logging.getLogger(__name__)


class SabreSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The heuristic aims to minimize the number
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
    physical qubits in the neighborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    This transpiler pass adds onto the SABRE algorithm in that it will run
    multiple trials of the algorithm with different seeds. The best output,
    determined by the trial with the least amount of SWAPed inserted, will
    be selected from the random trials.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, heuristic="basic", seed=None, fake_run=False, trials=None):
        r"""SabreSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
            trials (int): The number of seed trials to run sabre with. These will
                be run in parallel (unless the PassManager is already running in
                parallel). If not specified this defaults to the number of physical
                CPUs on the local system. For reproducible results it is recommended
                that you set this explicitly, as the output will be deterministic for
                a fixed number of trials.

        Raises:
            TranspilerError: If the specified heuristic is not valid.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'lookahead':

            This is the sum of two costs: first is the same as the basic cost.
            Second is the basic cost but now evaluated for the
            extended set as well (i.e. :math:`|E|` number of upcoming successors to gates in
            front_layer F). This is weighted by some amount EXTENDED_SET_WEIGHT (W) to
            signify that upcoming gates are less important that the front_layer.

            .. math::

                H_{decay}=\frac{1}{\left|{F}\right|}\sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
                    + W*\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'decay':

            This is the same as 'lookahead', but the whole cost is multiplied by a
            decay factor. This increases the cost if the SWAP that generated the
            trial layout was recently used (i.e. it penalizes increase in depth).

            .. math::

                H_{decay} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
                    \frac{1}{\left|{F}\right|} \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]\\
                    + W *\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]
                    }
        """

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
            self.target = None
        if self.coupling_map is not None and not self.coupling_map.is_symmetric:
            # A deepcopy is needed here if we don't own the coupling map (i.e. we were given it,
            # rather than calculated it from the Target), to avoid modifications updating shared
            # references in passes which require directional constraints.
            if isinstance(coupling_map, CouplingMap):
                self.coupling_map = deepcopy(self.coupling_map)
            self.coupling_map.make_symmetric()
        self._neighbor_table = None
        if self.coupling_map is not None:
            self._neighbor_table = NeighborTable(
                rustworkx.adjacency_matrix(self.coupling_map.graph)
            )

        self.heuristic = heuristic
        self.seed = seed
        if trials is None:
            self.trials = CPU_COUNT
        else:
            self.trials = trials

        self.fake_run = fake_run
        self._qubit_indices = None
        self._clbit_indices = None
        self.dist_matrix = None

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """

        if self.coupling_map is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        num_dag_qubits = len(dag.qubits)
        num_coupling_qubits = self.coupling_map.size()
        if num_dag_qubits < num_coupling_qubits:
            raise TranspilerError(
                f"Fewer qubits in the circuit ({num_dag_qubits}) than the coupling map"
                f" ({num_coupling_qubits})."
                " Have you run a layout pass and then expanded your DAG with ancillas?"
                " See `FullAncillaAllocation`, `EnlargeWithAncilla` and `ApplyLayout`."
            )
        if num_dag_qubits > num_coupling_qubits:
            raise TranspilerError(
                f"More qubits in the circuit ({num_dag_qubits}) than available in the coupling map"
                f" ({num_coupling_qubits})."
                " This circuit cannot be routed to this device."
            )

        if self.heuristic == "basic":
            heuristic = Heuristic.Basic
        elif self.heuristic == "lookahead":
            heuristic = Heuristic.Lookahead
        elif self.heuristic == "decay":
            heuristic = Heuristic.Decay
        else:
            raise TranspilerError("Heuristic %s not recognized." % self.heuristic)
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )

        self.dist_matrix = self.coupling_map.distance_matrix

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        self._qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        layout_mapping = {
            self._qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
        }
        layout = NLayout(layout_mapping, len(dag.qubits), self.coupling_map.size())
        original_layout = layout.copy()

        sabre_dag, circuit_to_dag_dict = _build_sabre_dag(
            dag,
            self.coupling_map.size(),
            self._qubit_indices,
        )
        sabre_result = build_swap_map(
            len(dag.qubits),
            sabre_dag,
            self._neighbor_table,
            self.dist_matrix,
            heuristic,
            layout,
            self.trials,
            self.seed,
        )

        layout_mapping = layout.layout_mapping()
        output_layout = Layout({dag.qubits[k]: v for (k, v) in layout_mapping})
        self.property_set["final_layout"] = output_layout
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()
            _apply_sabre_result(
                mapped_dag,
                dag,
                self._qubit_indices,
                canonical_register,
                original_layout,
                sabre_result,
                circuit_to_dag_dict,
            )
            return mapped_dag
        return dag


def _build_sabre_dag(dag, num_physical_qubits, qubit_indices):
    from qiskit.converters import circuit_to_dag

    # Maps id(block): circuit_to_dag(block) for all descendant blocks
    circuit_to_dag_dict = {}

    def recurse(block, block_qubit_indices):
        block_id = id(block)
        if block_id in circuit_to_dag_dict:
            block_dag = circuit_to_dag_dict[block_id]
        else:
            block_dag = circuit_to_dag(block)
            circuit_to_dag_dict[block_id] = block_dag
        return process_dag(block_dag, block_qubit_indices)

    def process_dag(block_dag, wire_map):
        dag_list = []
        node_blocks = {}
        for node in block_dag.topological_op_nodes():
            cargs_bits = set(node.cargs)
            if node.op.condition is not None:
                cargs_bits.update(condition_resources(node.op.condition).clbits)
            if isinstance(node.op, SwitchCaseOp):
                target = node.op.target
                if isinstance(target, Clbit):
                    cargs_bits.add(target)
                elif isinstance(target, ClassicalRegister):
                    cargs_bits.update(target)
                else:  # Expr
                    cargs_bits.update(node_resources(target).clbits)
            cargs = {block_dag.find_bit(x).index for x in cargs_bits}
            if isinstance(node.op, ControlFlowOp):
                node_blocks[node._node_id] = [
                    recurse(
                        block,
                        {inner: wire_map[outer] for inner, outer in zip(block.qubits, node.qargs)},
                    )
                    for block in node.op.blocks
                ]
            dag_list.append(
                (
                    node._node_id,
                    [wire_map[x] for x in node.qargs],
                    cargs,
                )
            )
        return SabreDAG(num_physical_qubits, block_dag.num_clbits(), dag_list, node_blocks)

    return process_dag(dag, qubit_indices), circuit_to_dag_dict


def _apply_sabre_result(
    mapped_dag,
    root_dag,
    qubit_indices,
    canonical_register,
    initial_layout,
    sabre_result,
    circuit_to_dag_dict,
    component_map=None,
):
    bit_to_qreg_idx = {bit: idx for idx, bit in enumerate(canonical_register)}

    def empty_dag(node, block):
        out = DAGCircuit()
        for qreg in mapped_dag.qregs.values():
            out.add_qreg(qreg)
        out.add_clbits(node.cargs)
        for creg in block.cregs:
            out.add_creg(creg)
        out._global_phase = block.global_phase
        return out

    def apply_inner(out_dag, current_layout, qubit_indices_inner, result, id_to_node):
        for node_id in result.node_order:
            node = id_to_node[node_id]
            if isinstance(node.op, ControlFlowOp):
                # Handle control flow op and continue.
                block_results = result.node_block_results[node_id]
                mapped_block_dags = []
                idle_qubits = set(out_dag.qubits)
                for block, block_result in zip(node.op.blocks, block_results):
                    block_id_to_node = circuit_to_dag_dict[id(block)]._multi_graph
                    mapped_block_dag = empty_dag(node, block)
                    mapped_block_layout = current_layout.copy()
                    block_qubit_indices = {
                        inner: qubit_indices_inner[outer]
                        for inner, outer in zip(block.qubits, node.qargs)
                    }
                    apply_inner(
                        mapped_block_dag,
                        mapped_block_layout,
                        block_qubit_indices,
                        block_result.result,
                        block_id_to_node,
                    )

                    # Apply swap epilogue to bring each block to the same
                    # final layout.
                    process_swaps(
                        block_result.swap_epilogue,
                        mapped_block_dag,
                        mapped_block_layout,
                        canonical_register,
                        False,
                        bit_to_qreg_idx,
                        component_map,
                    )

                    # If the swap epilogue didn't return us to the initial layout,
                    # there's a bug.
                    # assert mapped_block_layout.layout_mapping() == current_layout.layout_mapping()

                    mapped_block_dags.append(mapped_block_dag)
                    idle_qubits.intersection_update(mapped_block_dag.idle_wires())

                mapped_blocks = []
                for mapped_block_dag in mapped_block_dags:
                    # Remove wires that are idle in all blocks.
                    mapped_block_dag.remove_qubits(*idle_qubits)
                    mapped_blocks.append(dag_to_circuit(mapped_block_dag))

                # Apply the control flow gate to the dag.
                mapped_node = node.op.replace_blocks(mapped_blocks)
                mapped_node_qargs = mapped_blocks[0].qubits
                out_dag.apply_operation_back(mapped_node, mapped_node_qargs, node.cargs)
                continue

            # If we get here, the node isn't a control-flow gate.
            if node_id in result.map:
                process_swaps(
                    result.map[node_id],
                    out_dag,
                    current_layout,
                    canonical_register,
                    False,
                    bit_to_qreg_idx,
                    component_map,
                )
            apply_gate(
                out_dag,
                node,
                current_layout,
                canonical_register,
                False,
                qubit_indices_inner,
            )

    apply_inner(mapped_dag, initial_layout, qubit_indices, sabre_result, root_dag._multi_graph)


def process_swaps(
    swap_list,
    mapped_dag,
    current_layout,
    canonical_register,
    fake_run,
    qubit_indices,
    component_map=None,
):
    """
    Applies each swap in ``swap_list`` sequentially and updates
    ``current_layout`` accordingly.
    """
    for swap in swap_list:
        if component_map:
            swap_qargs = [
                canonical_register[component_map[swap[0]]],
                canonical_register[component_map[swap[1]]],
            ]
        else:
            swap_qargs = [canonical_register[swap[0]], canonical_register[swap[1]]]
        apply_gate(
            mapped_dag,
            DAGOpNode(op=SwapGate(), qargs=swap_qargs),
            current_layout,
            canonical_register,
            fake_run,
            qubit_indices,
        )
        if component_map:
            current_layout.swap_logical(component_map[swap[0]], component_map[swap[1]])
        else:
            current_layout.swap_logical(*swap)


def apply_gate(mapped_dag, node, current_layout, canonical_register, fake_run, qubit_indices):
    """Apply gate given the current layout."""
    new_node = transform_gate_for_layout(node, current_layout, canonical_register, qubit_indices)
    if fake_run:
        return new_node
    return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)


def transform_gate_for_layout(op_node, layout, device_qreg, qubit_indices):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = tuple(
        device_qreg[layout.logical_to_physical(qubit_indices[x])] for x in op_node.qargs
    )
    return mapped_op_node
