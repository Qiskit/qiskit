from qiskit.circuit import QuantumCircuit, CircuitInstruction, Gate
from qiskit.circuit.library import RealAmplitudes, LinearFunction
from qiskit.converters import circuit_to_dag, dag_to_dagdependency, dag_to_circuit, dagdependency_to_circuit, dagdependency_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization.collect_blocks import BlockCollector
from qiskit.circuit.random import random_circuit
from qiskit.transpiler.passes import Collect1qRuns
from test.python.quantum_info.operators.symplectic.test_clifford import random_clifford_circuit
import time


def _split_blocks_with_barriers_circuit(dag, all_blocks):
    """
    A hack (copy-pasting code from dag_to_circuit) that creates circuit from dag,
    based on the division of dag's nodes into blocks, that simply puts barriers
    after every block.
    """
    name = dag.name or None
    circuit = QuantumCircuit(
        dag.qubits,
        dag.clbits,
        *dag.qregs.values(),
        *dag.cregs.values(),
        name=name,
        global_phase=dag.global_phase,
    )
    circuit.metadata = dag.metadata
    circuit.calibrations = dag.calibrations

    for block in all_blocks:
        for node in block:
            circuit._append(CircuitInstruction(node.op.copy(), node.qargs, node.cargs))
        circuit.barrier()

    circuit.duration = dag.duration
    circuit.unit = dag.unit
    return circuit


def _replace_linear_block_with_op(dag, block):
    # Replace every discovered block by a linear function
    global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}

    # Create linear functions only out of blocks with at least 2 gates
    if len(block) == 1:
        return

    # Find the set of all qubits used in this block
    cur_qubits = set()
    for node in block:
        cur_qubits.update(node.qargs)

    # For reproducibility, order these qubits compatibly with the global order
    sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
    wire_pos_map = dict((qb, ix) for ix, qb in enumerate(sorted_qubits))

    # Construct a linear circuit
    qc = QuantumCircuit(len(cur_qubits))
    for node in block:
        if node.op.name == "cx":
            qc.cx(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])
        elif node.op.name == "swap":
            qc.swap(wire_pos_map[node.qargs[0]], wire_pos_map[node.qargs[1]])

    # Create a linear function from this quantum circuit
    op = LinearFunction(qc)

    # Replace the block by the constructed circuit
    dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)


def test_collect_matching_blocks(circuit, filter_fn, do_commutative_analysis, print_circuits=False):
    print(f"Original circuit: has #gates = {len(circuit.data)}, depth = {circuit.depth()}")
    if print_circuits:
        print("Original Circuit:")
        print(circuit)

    dag = circuit_to_dag(circuit)
    if do_commutative_analysis:
        dag = dag_to_dagdependency(dag)

    # dag.print()

    block_collector = BlockCollector(dag)
    matching_blocks, all_blocks = block_collector.collect_all_matching_blocks(filter_fn=filter_fn)

    for block in matching_blocks:
        _replace_linear_block_with_op(dag, block)

    # print(f"AFTER")
    # dag.print()


    circuit2 = dagdependency_to_circuit(dag)


    if do_commutative_analysis:
        circuit2 = dagdependency_to_circuit(dag)
    else:
        circuit2 = dag_to_circuit(dag)

    # circuit2 = dag_to_circuit(dag)

    # circuit2 = _split_blocks_with_barriers_circuit(dag, all_blocks)
    # # print(f"Num matching blocks: {len(matching_blocks)}, num all blocks: {len(all_blocks)}")
    # # print(f"Matching blocks have lengths: {[len(b) for b in matching_blocks]}")

    if print_circuits:
        print("Block Circuit:")
        print(circuit2)


def test_collect_commuting_blocks(circuit, filter_fn, do_commutative_analysis, print_circuits=False):
    print(f"Original circuit: has #gates = {len(circuit.data)}, depth = {circuit.depth()}")
    if print_circuits:
        print("Original Circuit:")
        print(circuit)

    dag = circuit_to_dag(circuit)
    if do_commutative_analysis:
        dag = dag_to_dagdependency(dag)

    block_collector = BlockCollector(dag)
    matching_blocks, all_blocks = block_collector.collect_all_commuting_blocks(filter_fn=filter_fn)

    circuit2 = _split_blocks_with_barriers_circuit(dag, all_blocks)
    print(f"Num commuting blocks: {len(matching_blocks)}, num all blocks: {len(all_blocks)}")
    # print(f"Commuting blocks have lengths: {[len(b) for b in matching_blocks]}")

    if print_circuits:
        print("Block Circuit:")
        print(circuit2)


def test_collect_parallel_blocks(circuit, do_commutative_analysis, print_circuits=False):
    print(f"Original circuit: has #gates = {len(circuit.data)}, depth = {circuit.depth()}")
    if print_circuits:
        print("Original Circuit:")
        print(circuit)

    from qiskit.transpiler.passes.optimization import OptimizeDepth

    circuit2 = PassManager(OptimizeDepth()).run(circuit)

    print(f"Final circuit: has #gates = {len(circuit2.data)}, depth = {circuit2.depth()}")
    if print_circuits:
        print("Block Circuit:")
        print(circuit2)


# Different filtering functions

def _is_linear(node):
    """Linear gates."""
    return node.op.name in ("cx", "swap") and node.op.condition is None


def _is_clifford(node):
    """Linear gates."""
    return node.op.name in ("x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap") and node.op.condition is None


def _is_2q(node):
    """Two-qubit gates."""
    return isinstance(node.op, Gate) and node.op.condition is None and len(node.qargs) == 2


def _is_1q(node):
    """Single-qubit gates"""
    return isinstance(node.op, Gate) and node.op.condition is None and len(node.qargs) == 1


# Actual examples

def test_collect_linear_blocks_from_real_amplitudes():
    print("test_collect_linear_blocks_from_real_amplitudes:")
    ansatz = RealAmplitudes(4, reps=2)
    circuit = ansatz.decompose()
    # test_collect_matching_blocks(circuit, _is_linear, do_commutative_analysis=False, print_circuits=True)
    test_collect_matching_blocks(circuit, _is_linear, do_commutative_analysis=True, print_circuits=True)


def test_collect_linear_blocks_exploring_commutativity():
    print("test_collect_linear_blocks_exploring_commutativity:")
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.z(0)
    qc.cx(0, 3)
    qc.cx(0, 4)
    # test_collect_matching_blocks(qc, _is_linear, do_commutative_analysis=False, print_circuits=True)
    test_collect_matching_blocks(qc, _is_linear, do_commutative_analysis=True, print_circuits=True)


def test_commuting_2q_gate_grouper():
    print("test_commuting_2q_gate_grouper:")
    from qiskit.circuit.library import TwoLocal
    circuit = TwoLocal(5, "ry", "cz", entanglement='pairwise').decompose()
    # test_collect_commuting_blocks(circuit, _is_2q, do_commutative_analysis=False, print_circuits=True)
    test_collect_commuting_blocks(circuit, _is_2q, do_commutative_analysis=True, print_circuits=True)


def test_collect_linear_blocks_random_circuit():
    print("test_collect_linear_blocks_random_circuit:")
    for seed in range(1):
        qc = random_circuit(5, 100, max_operands=2, seed=seed)
        test_collect_matching_blocks(qc, _is_linear, do_commutative_analysis=False, print_circuits=False)
        test_collect_matching_blocks(qc, _is_linear, do_commutative_analysis=True, print_circuits=False)
        print("")


def test_collect_clifford_blocks_random_circuit():
    print("test_collect_clifford_blocks_random_circuit:")
    for seed in range(10):
        qc = random_circuit(5, 100, max_operands=2, seed=seed)
        test_collect_matching_blocks(qc, _is_clifford, do_commutative_analysis=False, print_circuits=False)
        test_collect_matching_blocks(qc, _is_clifford, do_commutative_analysis=True, print_circuits=False)
        print("")


def test_random_circuit_depth_min():
    print("test_random_circuit_depth_min:")
    # it seems that random_circuits come quite depth-optimized, so the technique performs worse
    for seed in range(10):
        qc = random_circuit(5, 100, max_operands=2, seed=seed)
        # test_collect_parallel_blocks(qc, do_commutative_analysis=False, print_circuits=False)
        test_collect_parallel_blocks(qc, do_commutative_analysis=True, print_circuits=False)
        print("")


def test_random_clifford_circuit_depth_min():
    print("test_random_clifford_circuit_depth_min:")
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    for seed in range(10):
        qc = random_clifford_circuit(5, 100, gates, seed=seed)
        # test_collect_parallel_blocks(qc, do_commutative_analysis=False, print_circuits=False)
        test_collect_parallel_blocks(qc, do_commutative_analysis=True, print_circuits=False)
        print("")


if __name__ == "__main__":
    start_time = time.time()
    # test_collect_linear_blocks_from_real_amplitudes()
    # test_collect_linear_blocks_exploring_commutativity()
    # test_commuting_2q_gate_grouper()
    # test_collect_linear_blocks_random_circuit()
    # test_collect_clifford_blocks_random_circuit()
    # test_random_circuit_depth_min()
    test_random_clifford_circuit_depth_min()
    end_time = time.time()
    print(f"Total time = {end_time-start_time:.4}")
