import cProfile, pstats
import numpy as np
import time
import qiskit.circuit.library
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dagdependency
from qiskit.dagcircuit import DAGCircuit, DAGDependency
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation, InverseCancellation, \
    TemplateOptimization
from qiskit.transpiler.passes.optimization.commutative_inverse_cancellation import CommutativeInverseCancellation
from test.python.quantum_info.operators.symplectic.test_clifford import random_clifford_circuit
from qiskit.circuit.library import Permutation, LinearFunction, CXGate, HGate, SGate, SdgGate, SwapGate, CZGate, ZGate, XGate, YGate
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators import Clifford
from qiskit.circuit.library.templates import *


# =====================================
# CIRCUIT OPTIMIZATION
# =====================================

def optimize_circuit_aux(qc, is_cliff=True):
    # print("=====> RUNNING CIRCUIT OPTIMIZATION  <=====")

    gates_to_cancel = [CXGate(), HGate(), SwapGate(), CZGate(), ZGate(), XGate(), YGate()]

    ops = qc.count_ops()
    print(f"Before: #gates = {sum([ops[x] for x in ops])}")
    # print(f"ops: {ops}")
    # print(f"#gates = {sum([ops[x] for x in ops])}")
    if is_cliff:
        cliff = Clifford(qc)
        # print(cliff)
    # print("")

    # print("=== Transpiling ===")
    start_time = time.time()
    qc2 = transpile(qc, optimization_level=3)
    end_time = time.time()
    ops2 = qc2.count_ops()
    # print(f"ops2: {ops2}")
    # print(f"#gates = {sum([ops2[x] for x in ops2])}")
    # print(f"time = {end_time-start_time:.4f}")
    print(f"Transpile: #qubits = {qc.num_qubits}, #gates = {sum([ops[x] for x in ops])}, #gates_opt = {sum([ops2[x] for x in ops2])}, time = {end_time - start_time:.4f}")

    # if is_cliff:
    #     cliff2 = Clifford(qc2)
    #     print(cliff2)
    #     ok = cliff == cliff2
    #     print(f"{ok = }")
    #     assert ok
    # print("")

    # print("=== InverseCancellation ===")
    start_time = time.time()
    pm2 = PassManager(InverseCancellation(gates_to_cancel=gates_to_cancel))
    qc2 = pm2.run(qc)
    end_time = time.time()
    ops2 = qc2.count_ops()
    # print(f"ops2: {ops2}")
    # print(f"#gates = {sum([ops2[x] for x in ops2])}")
    # print(f"time = {end_time-start_time:.4f}")
    print(f"InverseCancellation: #qubits = {qc.num_qubits}, #gates = {sum([ops[x] for x in ops])}, #gates_opt = {sum([ops2[x] for x in ops2])}, time = {end_time - start_time:.4f}")

    if is_cliff:
        cliff2 = Clifford(qc2)
        # print(cliff2)
        ok = cliff == cliff2
        # print(f"{ok = }")
        assert ok
    # print("")

    # print("=== CommutativeInverseCancellation ===")
    start_time = time.time()
    pm2 = PassManager(CommutativeInverseCancellation())
    qc2 = pm2.run(qc)
    end_time = time.time()
    ops2 = qc2.count_ops()
    # print(f"ops2: {ops2}")
    # print(f"#gates = {sum([ops2[x] for x in ops2])}")
    # print(f"time = {end_time-start_time:.4f}")
    print(f"CommutativeInverseCancellation: #qubits = {qc.num_qubits}, #gates = {sum([ops[x] for x in ops])}, #gates_opt = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")

    if is_cliff:
        cliff2 = Clifford(qc2)
        # print(cliff2)
        ok = cliff == cliff2
        # print(f"{ok = }")
        assert ok
    # print("")

    # print("=== CommutativeCancellation ===")
    start_time = time.time()
    pm2 = PassManager(CommutativeCancellation())
    qc2 = pm2.run(qc)
    end_time = time.time()
    ops2 = qc2.count_ops()
    # print(f"ops2: {ops2}")
    # print(f"#gates = {sum([ops2[x] for x in ops2])}")
    # print(f"time = {end_time-start_time:.4f}")
    print(f"CommutativeCancellation: #qubits = {qc.num_qubits}, #gates = {sum([ops[x] for x in ops])}, #gates_opt = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")

    # if is_cliff:
    #     cliff2 = Clifford(qc2)
    #     print(cliff2)
    #     ok = cliff == cliff2
    #     print(f"{ok = }")
    #     assert ok
    # print("")

    print("")


def optimize_aux(num_qubits, num_gates, seed):
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    qc = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed)
    optimize_circuit_aux(qc)


def optimize():
    optimize_aux(10, 1000, 0)
    optimize_aux(10, 5000, 0)
    optimize_aux(50, 1000, 0)
    optimize_aux(50, 5000, 0)



def optimize_circ1():
    print(f"==========RUNNING EXPERIMENT 1")
    qc = QuantumCircuit(2)
    qc.z(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.z(0)
    qc.x(1)
    print(qc)
    optimize_circuit_aux(qc)


def optimize_circ2():
    print(f"========RUNNING EXPERIMENT 2")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    print(qc)
    optimize_circuit_aux(qc)


def optimize_circ3():
    print(f"========RUNNING EXPERIMENT 3")
    qc = QuantumCircuit(2)
    qc.p(np.pi / 4, 0)
    qc.p(-np.pi / 4, 0)
    print(qc)
    optimize_circuit_aux(qc, is_cliff=False)


def optimize_circ4():
    print(f"========RUNNING EXPERIMENT 4")
    qc = QuantumCircuit(2)
    qc.p(np.pi / 4, 0)
    qc.p(-np.pi / 2, 0)
    print(qc)
    optimize_circuit_aux(qc, is_cliff=False)


def optimize_circ5():
    print(f"========RUNNING EXPERIMENT 5")
    qc = QuantumCircuit(3)
    qc.initialize("001", qc.qubits)
    qc.cx(0, 1)
    qc.measure_all()
    print(qc)
    qc2 = transpile(qc, optimization_level=0)
    print(qc2)
    # optimize_circuit_aux(qc, is_cliff=False)


# Only the new pass
def optimize_circuit_commutative_inverse_cancellation_aux(qc, is_cliff=True):
    # print("=== CommutativeInverseCancellation ===")
    gates_to_cancel = [CXGate(), HGate(), SwapGate(), CZGate(), ZGate(), XGate(), YGate()]
    if is_cliff:
        cliff = Clifford(qc)
    ops = qc.count_ops()

    start_time = time.time()
    pm2 = PassManager(CommutativeInverseCancellation())
    qc2 = pm2.run(qc)
    end_time = time.time()
    ops2 = qc2.count_ops()
    # print(f"ops2: {ops2}")
    # print(f"#gates = {sum([ops2[x] for x in ops2])}")
    # print(f"time = {end_time-start_time:.4f}")
    print(f"CommutativeInverseCancellation: #qubits = {qc.num_qubits}, #gates = {sum([ops[x] for x in ops])}, #gates_opt = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")
    if is_cliff:
        cliff2 = Clifford(qc2)
        # print(cliff2)
        ok = cliff == cliff2
        # print(f"{ok = }")
        assert ok
    # print("")


def optimize_commutative_inverse_cancellation_aux(num_qubits, num_gates, seed):
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    qc = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed)
    optimize_circuit_commutative_inverse_cancellation_aux(qc)


def optimize_commutative_inverse_cancellation():
    optimize_commutative_inverse_cancellation_aux(5, 1000, 0)
    optimize_commutative_inverse_cancellation_aux(5, 5000, 0)
    optimize_commutative_inverse_cancellation_aux(10, 1000, 0)
    optimize_commutative_inverse_cancellation_aux(10, 5000, 0)
    optimize_commutative_inverse_cancellation_aux(50, 1000, 0)
    optimize_commutative_inverse_cancellation_aux(50, 5000, 0)


def time_commutative_inverse_cancellation():
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    circuits = []
    for seed in range(10):
        qc = random_clifford_circuit(50, 5000, gates=gates, seed=seed)
        circuits.append(qc)

    pm2 = PassManager(CommutativeInverseCancellation())

    time_start = time.time()
    for qc in circuits:
        qc2 = pm2.run(qc)
    time_end = time.time()

    print(f"time_commutative_inverse: {time_end - time_start:.4f}")

# =====================================
# DAG DEPENDENCY
# =====================================

def time_construct_dag_dependency_aux(num_qubits, num_gates, seed):
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    qc = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed)
    start_time = time.time()
    dag_dependency = circuit_to_dagdependency(qc)
    end_time = time.time()
    print(f"DagDependency: #qubits = {num_qubits}, #gates = {num_gates}, seed = {seed}, #edges = {len(dag_dependency.get_all_edges())}, depth = {dag_dependency.depth()}, time = {end_time - start_time:.4f}")


def time_construct_dag_dependency():
    time_construct_dag_dependency_aux(5, 100, seed=0)
    time_construct_dag_dependency_aux(5, 1000, seed=0)
    time_construct_dag_dependency_aux(5, 5000, seed=0)
    time_construct_dag_dependency_aux(10, 100, seed=0)
    time_construct_dag_dependency_aux(10, 1000, seed=0)
    time_construct_dag_dependency_aux(10, 5000, seed=0)


from memory_profiler import profile

@profile
def memory_construct_dag_dependency_aux(num_qubits, num_gates, seed):
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    qc = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed)
    dag_dependency = circuit_to_dagdependency(qc)
    print(f"DagDependency: #qubits = {num_qubits}, #gates = {num_gates}, seed = {seed}, #edges = {len(dag_dependency.get_all_edges())}")


def memory_construct_dag_dependency():
    memory_construct_dag_dependency_aux(5, 100, seed=0)
    memory_construct_dag_dependency_aux(5, 1000, seed=0)
    memory_construct_dag_dependency_aux(5, 5000, seed=0)
    memory_construct_dag_dependency_aux(10, 100, seed=0)
    memory_construct_dag_dependency_aux(10, 1000, seed=0)
    memory_construct_dag_dependency_aux(10, 5000, seed=0)


# =====================================
# TEMPLATE OPTIMIZATION
# =====================================

def time_template_optimization_aux(num_qubits, num_gates, seed):
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    qc = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed)
    # ops = qc.count_ops()
    # print(f"#gates = {sum([ops[x] for x in ops])}")

    # All Clifford templates without S*H*S*H*S*H = Id, since this only holds *up to a phase*.
    template_list = [clifford_2_1(),
                     clifford_2_2(),
                     clifford_2_3(),
                     clifford_2_4(),
                     clifford_3_1(),
                     clifford_4_1(),
                     clifford_4_2(),
                     clifford_4_3(),
                     clifford_4_4(),
                     clifford_5_1(),
                     clifford_6_1(),
                     clifford_6_2(),
                     clifford_6_3(),
                     # clifford_6_4(),
                     clifford_6_5(),
                     clifford_8_1(),
                     clifford_8_2(),
                     clifford_8_3()]

    pm = PassManager(TemplateOptimization(template_list=template_list))
    start_time = time.time()
    qc2 = pm.run(qc)
    end_time = time.time()
    ops = qc2.count_ops()
    # print(f"#gates = {sum([ops[x] for x in ops])}")
    print(f"Templates: #qubits = {num_qubits}, #gates = {num_gates}, seed = {seed}, #gates_opt = {sum([ops[x] for x in ops])}, time = {end_time - start_time:.4f}")


def time_template_optimization():
    time_template_optimization_aux(5, 100, seed=0)
    time_template_optimization_aux(5, 1000, seed=0)
    time_template_optimization_aux(10, 100, seed=0)
    time_template_optimization_aux(10, 1000, seed=0)


def test_commutative1():
    qc1 = QuantumCircuit(5)
    qc1.z(3)
    qc1.cx(3, 0)

    qc2 = QuantumCircuit(5)
    qc2.cx(3, 0)
    qc2.z(3)

    print(Operator(qc1) == Operator(qc2))

def test_commutative2():
    qc1 = QuantumCircuit(5)
    qc1.cz(3, 1)
    qc1.sdg(1)

    qc2 = QuantumCircuit(5)
    qc2.sdg(1)
    qc2.cz(3, 1)

    print(Operator(qc1) == Operator(qc2))


def test_circ1():
    qc = QuantumCircuit(3)
    qc.z(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.z(0)
    qc.x(1)
    print(qc)
    pm2 = PassManager(CommutativeInverseCancellation())
    qc2 = pm2.run(qc)
    print(qc2)


def test_successors_predecessors():
    """Test the method direct_successors."""

    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.x(0)
    circuit.h(0)
    circuit.x(1)
    circuit.h(0)
    circuit.measure(0, 0)
    print(circuit)

    dag = circuit_to_dagdependency(circuit)


    dir_successors_second = dag.direct_successors(1)
    print(f"{dir_successors_second = }, should be [2, 4]")

    dir_successors_fourth = dag.direct_successors(3)
    print(f"{dir_successors_fourth = }, should be []")

    successors_second = dag.successors(1)
    print(f"{successors_second = }, should be [2, 4, 5]")

    successors_fourth = dag.successors(3)
    print(f"{successors_fourth = }, should be []")

    dir_predecessors_sixth = dag.direct_predecessors(5)
    print(f"{dir_predecessors_sixth = }, should be [2, 4]")

    dir_predecessors_fourth = dag.direct_predecessors(3)
    print(f"{dir_predecessors_fourth = }, should be []")

    predecessors_sixth = dag.predecessors(5)
    print(f"{predecessors_sixth = }, should be [0, 1, 2, 4]")

    predecessors_fourth = dag.predecessors(3)
    print(f"{predecessors_fourth = }, should be []")


if __name__ == "__main__":
    time_construct_dag_dependency()
    # memory_construct_dag_dependency()
    # time_template_optimization()
    # optimize()
    optimize_commutative_inverse_cancellation()
    # optimize_circ1()
    # optimize_circ2()
    # optimize_circ3()
    # optimize_circ4()
    # optimize_circ5()
    # test_commutative2()
    # test_circ1()

    time_commutative_inverse_cancellation()

    # test_successors_predecessors()


