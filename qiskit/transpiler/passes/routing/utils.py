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

"""Utility functions for routing"""

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError


def route_cf_multiblock(routing_pass, node, current_layout, root_dag, seed=None):
    """Route control flow instructions which contain multiple blocks (e.g. :class:`.IfElseOp`).
    Since each control flow block may yield a different layout, this function applies swaps to the
    shorter depth blocks to make all final layouts match.

    Args:
        routing_pass (StochasticSwap):  An instance of the routing pass that should be called
            recursively to route inside each of the blocks.
        node (DAGOpNode): A DAG node whose operation is a :class:`.ControlFlowOp` that contains more
            than one block, such as :class:`.IfElseOp`.
        current_layout (Layout): The current layout at the start of the instruction.
        root_dag (DAGCircuit): root dag of compilation
        seed (int): seed for RNG of internal layout transformation.

    Returns:
        ControlFlowOp: routed control flow operation.
        final_layout (Layout): layout after instruction.
        list(Qubit): list of idle qubits in controlflow layer.
    """
    # pylint: disable=cyclic-import
    from qiskit.converters import dag_to_circuit, circuit_to_dag

    # For each block, expand it up be the full width of the containing DAG so we can be certain that
    # it is routable, then route it within that.  When we recombine later, we'll reduce all these
    # blocks down to remove any qubits that are idle.
    block_dags = []
    block_layouts = []
    indices = {bit: i for i, bit in enumerate(root_dag.qubits)}
    order = [indices[bit] for bit in node.qargs]
    for block in node.op.blocks:
        full_dag_block = root_dag.copy_empty_like()
        dag_block = circuit_to_dag(block)
        full_dag_block.compose(dag_block, qubits=order)
        routing_pass.initial_layout = current_layout
        updated_dag_block = routing_pass.run(full_dag_block)
        block_dags.append(updated_dag_block)
        block_layouts.append(routing_pass.property_set["final_layout"].copy())

    # Add swaps to the end of each block to make sure they all have the same layout at the end.  As
    # a heuristic we choose the final layout of the deepest block to be the target for everyone.
    # Adding these swaps can cause fewer wires to be idle than we expect (if we have to swap across
    # unused qubits), so we track that at this point too.
    deepest_index = np.argmax([block.depth(recurse=True) for block in block_dags])
    final_layout = block_layouts[deepest_index]
    p2v = current_layout.get_physical_bits()
    idle_qubits = set(root_dag.qubits)
    for i, updated_dag_block in enumerate(block_dags):
        if i != deepest_index:
            swap_circuit, swap_qubits = _get_swap_map_dag(
                root_dag,
                routing_pass.coupling_map,
                block_layouts[i],
                final_layout,
                seed=seed,
            )
            if swap_circuit.depth():
                virtual_swap_dag = updated_dag_block.copy_empty_like()
                order = [p2v[virtual_swap_dag.qubits.index(qubit)] for qubit in swap_qubits]
                virtual_swap_dag.compose(swap_circuit, qubits=order)
                updated_dag_block.compose(virtual_swap_dag)
        idle_qubits &= set(updated_dag_block.idle_wires())

    # Now for each block, expand it to be full width over all active wires (all blocks of a
    # control-flow operation need to have the same input wires), and convert it to the circuit form.
    block_circuits = []
    for i, updated_dag_block in enumerate(block_dags):
        updated_dag_block.remove_qubits(*idle_qubits)
        new_dag_block = DAGCircuit()
        new_num_qubits = updated_dag_block.num_qubits()
        qreg = QuantumRegister(new_num_qubits, "q")
        new_dag_block.add_qreg(qreg)
        for creg in updated_dag_block.cregs.values():
            new_dag_block.add_creg(creg)
        for inner_node in updated_dag_block.op_nodes():
            new_qargs = [qreg[updated_dag_block.qubits.index(bit)] for bit in inner_node.qargs]
            new_dag_block.apply_operation_back(inner_node.op, new_qargs, inner_node.cargs)
        block_circuits.append(dag_to_circuit(new_dag_block))

    return node.op.replace_blocks(block_circuits), final_layout, idle_qubits


def route_cf_looping(routing_pass, node, current_layout, root_dag, seed=None):
    """Route a control-flow operation that represents a loop, such as :class:`.ForOpLoop` or
    :class:`.WhileOpLoop`.  Importantly, these operations have a single block inside, and the final
    layout of the block needs to match the initial layout so the loop can continue.

    Args:
        routing_pass (StochasticSwap):  An instance of the routing pass that should be called
            recursively to route inside each of the blocks.
        node (DAGOpNode): A DAG node whose operation is a :class:`.ControlFlowOp` that represents a
            loop with a single block, such as :class:`.ForLoopOp`.
        current_layout (Layout): The current layout at the start of the instruction.
        root_dag (DAGCircuit): root dag of compilation
        seed (int): seed for RNG of internal layout transformation.

    Returns:
        ControlFlowOp: routed control flow operation.
        Layout: layout after instruction (this will be the same as the input layout).
        list(Qubit): list of idle qubits in controlflow layer.
    """
    # pylint: disable=cyclic-import
    from qiskit.converters import dag_to_circuit, circuit_to_dag

    # Temporarily expand to full width, and route within that.
    full_dag_block = root_dag.copy_empty_like()
    indices = {bit: i for i, bit in enumerate(root_dag.qubits)}
    order = [indices[bit] for bit in node.qargs]
    full_dag_block.compose(circuit_to_dag(node.op.blocks[0]), qubits=order)
    updated_dag_block = routing_pass.run(full_dag_block)

    # Ensure that the layout at the end of the block is returned to being the layout at the start of
    # the block again, so the loop works.
    updated_layout = routing_pass.property_set["final_layout"].copy()
    swap_circuit, swap_qubits = _get_swap_map_dag(
        root_dag, routing_pass.coupling_map, updated_layout, current_layout, seed=seed
    )
    if swap_circuit.depth():
        p2v = current_layout.get_physical_bits()
        virtual_swap_dag = updated_dag_block.copy_empty_like()
        order = [p2v[virtual_swap_dag.qubits.index(qubit)] for qubit in swap_qubits]
        virtual_swap_dag.compose(swap_circuit, qubits=order)
        updated_dag_block.compose(virtual_swap_dag)

    # Contract the routed block back down to only operate on the qubits that it actually needs.
    idle_qubits = set(root_dag.qubits) & set(updated_dag_block.idle_wires())
    updated_dag_block.remove_qubits(*idle_qubits)
    new_dag_block = DAGCircuit()
    new_num_qubits = updated_dag_block.num_qubits()
    qreg = QuantumRegister(new_num_qubits, "q")
    new_dag_block.add_qreg(qreg)
    for creg in updated_dag_block.cregs.values():
        new_dag_block.add_creg(creg)
    for inner_node in updated_dag_block.op_nodes():
        new_qargs = [qreg[updated_dag_block.qubits.index(bit)] for bit in inner_node.qargs]
        new_dag_block.apply_operation_back(inner_node.op, new_qargs, inner_node.cargs)
    updated_circ_block = dag_to_circuit(new_dag_block)
    node.op.num_qubits = updated_circ_block.num_qubits
    return node.op.replace_blocks([updated_circ_block]), current_layout, idle_qubits


def combine_permutations(*permutations):
    """
    Chain a series of permutations.

    Args:
        *permutations (list(int)): permutations to combine

    Returns:
        list: combined permutation
    """
    order = permutations[0]
    for this_order in permutations[1:]:
        order = [order[i] for i in this_order]
    return order


def _get_swap_map_dag(dag, coupling_map, from_layout, to_layout, seed, trials=4):
    """Gets the circuit of swaps to go from from_layout to to_layout."""
    from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper

    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        raise TranspilerError("layout transformation runs on physical circuits only")

    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError("The layout does not match the amount of qubits in the DAG")

    if coupling_map:
        graph = coupling_map.graph.to_undirected()
    else:
        coupling_map = CouplingMap.from_full(len(to_layout))
        graph = coupling_map.graph.to_undirected()

    token_swapper = ApproximateTokenSwapper(graph, seed)
    # Find the permutation between the initial physical qubits and final physical qubits.
    permutation = {
        pqubit: to_layout.get_virtual_bits()[vqubit]
        for vqubit, pqubit in from_layout.get_virtual_bits().items()
    }
    permutation_circ = token_swapper.permutation_circuit(permutation, trials)
    permutation_qubits = [
        dag.qubits[i[0]] for i in sorted(permutation_circ.inputmap.items(), key=lambda x: x[0])
    ]
    return permutation_circ.circuit, permutation_qubits
