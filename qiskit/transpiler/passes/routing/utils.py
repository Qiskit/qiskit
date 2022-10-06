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


def route_cf_multiblock(tpass, cf_opnode, current_layout, root_dag, seed=None):
    """Transpile control flow instructions which may contain multiple
    blocks (e.g. IfElseOp). Since each control flow block may yield a
    different layout, this function applies swaps to the shorter depth
    blocks to make all final layouts match.

    Args:
        tpass (BasePass): Transpiler pass object to use recursively.
        cf_opnode (DAGOpNode): multiblock instruction node e.g. IfElseOp.
        current_layout (Layout): The current layout at the start of the instruction.
        root_dag (DAGCircuit): root dag of compilation
        seed (int): seed for RNG of internal layout transformation.

    Returns:
        IfElseOp: transpiled control flow operation
        final_layout (Layout): layout after instruction
        list(Qubit): list of idle qubits in controlflow layer.
    """
    # pylint: disable=cyclic-import
    from qiskit.converters import dag_to_circuit, circuit_to_dag

    coupling = tpass.coupling_map
    block_dags = []  # control flow dag blocks
    block_layouts = []  # control flow layouts
    # expand to full width for routing
    cf_op = cf_opnode.op
    indices = {bit: i for i, bit in enumerate(root_dag.qubits)}
    order = [indices[bit] for bit in cf_opnode.qargs]
    for block in cf_op.blocks:
        full_dag_block = root_dag.copy_empty_like()
        dag_block = circuit_to_dag(block)
        full_dag_block.compose(dag_block, qubits=order)
        tpass.initial_layout = current_layout
        updated_dag_block = tpass.run(full_dag_block)
        block_dags.append(updated_dag_block)
        block_layouts.append(tpass.property_set["final_layout"].copy())
    deepest_index = np.argmax([block.depth(recurse=True) for block in block_dags])
    block_circuits = [None] * len(block_layouts)
    p2v = current_layout.get_physical_bits()
    idle_qubits = set(root_dag.qubits)
    for i, updated_dag_block in enumerate(block_dags):
        if i == deepest_index:
            block_circuits[i] = dag_to_circuit(updated_dag_block)
        else:
            swap_circuit, swap_qubits = _get_swap_map_dag(
                root_dag, coupling, block_layouts[i], block_layouts[deepest_index], seed=seed
            )
            if swap_circuit.depth():
                virtual_swap_dag = updated_dag_block.copy_empty_like()
                order = [p2v[virtual_swap_dag.qubits.index(qubit)] for qubit in swap_qubits]
                virtual_swap_dag.compose(swap_circuit, qubits=order)
                updated_dag_block.compose(virtual_swap_dag)
        idle_qubits &= set(updated_dag_block.idle_wires())
    # contract idle bits from full width post routing
    for i, updated_dag_block in enumerate(block_dags):
        updated_dag_block.remove_qubits(*idle_qubits)
        new_dag_block = DAGCircuit()
        new_num_qubits = updated_dag_block.num_qubits()
        qreg = QuantumRegister(new_num_qubits, "q")
        new_dag_block.add_qreg(qreg)
        for creg in updated_dag_block.cregs.values():
            new_dag_block.add_creg(creg)
        for node in updated_dag_block.op_nodes():
            new_qargs = [qreg[updated_dag_block.qubits.index(bit)] for bit in node.qargs]
            new_dag_block.apply_operation_back(node.op, new_qargs, node.cargs)
        block_circuits[i] = dag_to_circuit(new_dag_block)

    final_layout = block_layouts[deepest_index]
    return cf_op.replace_blocks(block_circuits), final_layout, idle_qubits


def route_cf_looping(tpass, cf_opnode, current_layout, root_dag, seed=None):
    """For looping this pass adds a swap layer using ApproximateTokenSwapper
    to the end of the loop body to bring the layout back to the
    starting layout. This prevents reapplying layout changing
    swaps for every iteration of the loop.

    Args:
        tpass (BasePass): pass object to run
        cf_opnode (DAGOpNode): looping instruction e.g. ForLoopOp, WhileLoopOp
        current_layout (Layout): The current layout at the start and by the
           end of the instruction.
        root_dag (DAGCircuit): root dagcircuit
        seed (int): seed for RNG of internal layout transformation.

    Returns:
        IfElseOp: transpiled control flow operation
        final_layout (Layout): layout after instruction
        list(Qubit): list of idle qubits in controlflow layer.
    """
    # pylint: disable=cyclic-import
    from qiskit.converters import dag_to_circuit, circuit_to_dag

    cf_op = cf_opnode.op  # control flow operation
    coupling = tpass.coupling_map
    dag_block = circuit_to_dag(cf_op.blocks[0])
    # expand to full width for routing
    full_dag_block = root_dag.copy_empty_like()
    start_layout = current_layout
    indices = {bit: i for i, bit in enumerate(root_dag.qubits)}
    order = [indices[bit] for bit in cf_opnode.qargs]
    full_dag_block.compose(dag_block, qubits=order)
    updated_dag_block = tpass.run(full_dag_block)
    updated_layout = tpass.property_set["final_layout"].copy()
    swap_circuit, swap_qubits = _get_swap_map_dag(
        root_dag, coupling, updated_layout, start_layout, seed=seed
    )

    if swap_circuit.depth():
        p2v = current_layout.get_physical_bits()
        virtual_swap_dag = updated_dag_block.copy_empty_like()
        order = [p2v[virtual_swap_dag.qubits.index(qubit)] for qubit in swap_qubits]
        virtual_swap_dag.compose(swap_circuit, qubits=order)
        updated_dag_block.compose(virtual_swap_dag)
    # contract from full width post routing
    idle_qubits = set(root_dag.qubits) & set(updated_dag_block.idle_wires())
    updated_dag_block.remove_qubits(*idle_qubits)
    new_dag_block = DAGCircuit()
    new_num_qubits = updated_dag_block.num_qubits()
    qreg = QuantumRegister(new_num_qubits, "q")
    new_dag_block.add_qreg(qreg)
    for creg in updated_dag_block.cregs.values():
        new_dag_block.add_creg(creg)
    for node in updated_dag_block.op_nodes():
        new_qargs = [qreg[updated_dag_block.qubits.index(bit)] for bit in node.qargs]
        new_dag_block.apply_operation_back(node.op, new_qargs, node.cargs)
    updated_circ_block = dag_to_circuit(new_dag_block)
    cf_op.num_qubits = updated_circ_block.num_qubits
    return cf_op.replace_blocks([updated_circ_block]), current_layout, idle_qubits


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
