from qiskit.circuit import QuantumCircuit, CircuitInstruction, Gate
from qiskit.circuit.library import RealAmplitudes, LinearFunction
from qiskit.converters import circuit_to_dag, dag_to_dagdependency, dag_to_circuit, dagdependency_to_circuit, \
    dagdependency_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization.collect_blocks import BlockCollector
from qiskit.circuit.random import random_circuit
from qiskit.transpiler.passes import Collect1qRuns
from test.python.quantum_info.operators.symplectic.test_clifford import random_clifford_circuit
import time
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.transpiler.passes import CommutationAnalysis
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import Commuting2qGateGrouper, SwapStrategy
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import Commuting2qGateRouter
from qiskit.circuit.library import TwoLocal


def test_collect_linear_blocks(circuit, do_commutative_analysis, print_circuits=False):
    print(f"Original circuit: has gates = {circuit.count_ops()}, depth = {circuit.depth()}")
    if print_circuits:
        print(circuit)

    from qiskit.transpiler.passes.optimization import CollectLinearFunctions
    circuit2 = PassManager(CollectLinearFunctions(split_blocks=True,
                                                  do_commutative_analysis=do_commutative_analysis,
                                                  min_nodes_per_block=1)).run(circuit)

    print(f"Block circuit: has gates = {circuit2.count_ops()}, depth = {circuit.depth()}")

    if print_circuits:
        print(circuit2)


def test_collect_clifford_blocks(circuit, do_commutative_analysis, print_circuits=False):
    print(f"Original circuit: has gates = {circuit.count_ops()}, depth = {circuit.depth()}")
    if print_circuits:
        print(circuit)

    from qiskit.transpiler.passes.optimization import CollectCliffords
    circuit2 = PassManager(CollectCliffords(split_blocks=True,
                                            do_commutative_analysis=do_commutative_analysis,
                                            min_nodes_per_block=1)).run(circuit)

    print(f"Block circuit: has gates = {circuit2.count_ops()}, depth = {circuit.depth()}")

    if print_circuits:
        print(circuit2)


def test_collect_parallel_blocks(circuit, do_commutative_analysis, print_circuits=False):
    print(f"Original circuit: has #gates = {len(circuit.data)}, depth = {circuit.depth()}")
    if print_circuits:
        print(circuit)

    from qiskit.transpiler.passes.optimization import OptimizeDepth

    circuit2 = PassManager(OptimizeDepth()).run(circuit)

    print(f"Final circuit: has #gates = {len(circuit2.data)}, depth = {circuit2.depth()}")
    if print_circuits:
        print(circuit2)


def test_collect_commuting_2q_blocks(circuit, split_blocks, min_nodes_per_block, print_circuits=False):
    print(f"Original circuit: has #gates = {len(circuit.data)}, depth = {circuit.depth()}")
    if print_circuits:
        print(circuit)

    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import Commuting2qGateGrouper
    circuit2 = PassManager(
        Commuting2qGateGrouper(split_blocks=split_blocks, min_nodes_per_block=min_nodes_per_block)).run(circuit)

    print(f"Final Circuit: has #gates = {len(circuit2.data)}, depth = {circuit2.depth()}")
    if print_circuits:
        print(circuit2)


# Actual examples

def test_collect_linear_blocks_from_real_amplitudes():
    print("test_collect_linear_blocks_from_real_amplitudes:")
    ansatz = RealAmplitudes(4, reps=2)
    circuit = ansatz.decompose()
    test_collect_linear_blocks(circuit, do_commutative_analysis=False, print_circuits=True)
    test_collect_linear_blocks(circuit, do_commutative_analysis=True, print_circuits=True)


def test_collect_linear_blocks_exploring_commutativity():
    print("test_collect_linear_blocks_exploring_commutativity:")
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.z(0)
    qc.cx(0, 3)
    qc.cx(0, 4)
    test_collect_linear_blocks(qc, do_commutative_analysis=False, print_circuits=True)
    test_collect_linear_blocks(qc, do_commutative_analysis=True, print_circuits=True)


def test_collect_linear_blocks_split():
    print("test_collect_linear_blocks_split:")
    qc = QuantumCircuit(6)
    qc.cx(0, 1)
    qc.swap(3, 5)
    qc.cx(2, 1)
    qc.cx(5, 3)
    test_collect_linear_blocks(qc, do_commutative_analysis=False, print_circuits=True)
    test_collect_linear_blocks(qc, do_commutative_analysis=True, print_circuits=True)


def test_collect_linear_blocks_random_circuit():
    print("test_collect_linear_blocks_random_circuit:")
    for seed in range(1):
        qc = random_circuit(5, 100, max_operands=2, seed=seed)
        test_collect_linear_blocks(qc, do_commutative_analysis=False, print_circuits=False)
        test_collect_linear_blocks(qc, do_commutative_analysis=True, print_circuits=False)
        print("")


def test_collect_clifford_blocks_random_circuit():
    print("test_collect_clifford_blocks_random_circuit:")
    for seed in range(10):
        qc = random_circuit(5, 100, max_operands=2, seed=seed)
        test_collect_clifford_blocks(qc, do_commutative_analysis=False, print_circuits=False)
        test_collect_clifford_blocks(qc, do_commutative_analysis=True, print_circuits=False)
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


# EXPERIMENTS WITH COMMUTING GATES

def test_commuting_2q_gate_grouper_example1():
    print("test_commuting_2q_gate_grouper_example1:")
    from qiskit.circuit.library import TwoLocal
    circuit = TwoLocal(5, "ry", "cz", entanglement='pairwise').decompose()
    test_collect_commuting_2q_blocks(circuit, split_blocks=True, min_nodes_per_block=2, print_circuits=True)


def test_commuting_2q_gate_grouper_example2():
    print("test_commuting_2q_gate_grouper_example2:")
    circuit = TwoLocal(5, "ry", "cz", entanglement='full', reps=2).decompose()
    print("Original circuit:")
    print(circuit)
    swap_strat = SwapStrategy.from_line([0, 1, 2, 3, 4])
    backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6)])
    initial_layout = Layout.from_intlist([0, 1, 3, 4, 5], *circuit.qregs)

    passmanager = PassManager(
        [
            CommutationAnalysis(),
            Commuting2qGateGrouper(),
            Commuting2qGateRouter(swap_strat),
            SetLayout(initial_layout),
            FullAncillaAllocation(backend_cmap),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ]
    )
    print("Transpiled circuit:")

    result = passmanager.run(circuit)
    print(result)



def test_commuting_2q_gate_grouper_example3():
    print("test_commuting_2q_gate_grouper_example3:")

    circ = QuantumCircuit(8)
    circ.cz(0, 1)
    circ.cz(2, 3)
    circ.cz(0, 2)
    circ.cz(1, 3)
    circ.cz(0, 3)

    circ.cz(5, 6)
    circ.cz(6, 7)
    circ.cz(5, 7)

    print(circ)

    passmanager = PassManager([Commuting2qGateGrouper(split_blocks=True, min_nodes_per_block=2)])
    res = passmanager.run(circ)
    print(res)




def test_commuting_2q_gate_grouper_example4():
    print("test_commuting_2q_gate_grouper_example4:")

    circ = QuantumCircuit(4)
    circ.cx(0, 1)
    circ.cx(1, 2)

    cmap = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
    cmap.make_symmetric()

    print(circ)

    passmanager = PassManager([Commuting2qGateGrouper(min_nodes_per_block=2)])
    res = passmanager.run(circ)
    print(res)


if __name__ == "__main__":
    start_time = time.time()
    # test_collect_linear_blocks_from_real_amplitudes()
    # test_collect_linear_blocks_exploring_commutativity()
    # test_collect_linear_blocks_split()
    # test_collect_linear_blocks_random_circuit()

    # test_collect_clifford_blocks_random_circuit()

    # test_random_circuit_depth_min()
    # test_random_clifford_circuit_depth_min()

    test_commuting_2q_gate_grouper_example1()
    test_commuting_2q_gate_grouper_example2()
    test_commuting_2q_gate_grouper_example3()
    test_commuting_2q_gate_grouper_example4()

    end_time = time.time()
    print(f"Total time = {end_time - start_time:.4}")
