import cProfile, pstats
import time
import numpy as np
from functools import lru_cache

import qiskit.circuit.library
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dagdependency
from qiskit.dagcircuit import DAGCircuit, DAGDependency
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation, InverseCancellation
from qiskit.transpiler.passes.optimization.commutative_inverse_cancellation import CommutativeInverseCancellation
from test.python.quantum_info.operators.symplectic.test_clifford import random_clifford_circuit

from qiskit.dagcircuit.dagdependency import _does_commute
from qiskit.circuit.library import Permutation, LinearFunction, CXGate, HGate, SGate, SdgGate, SwapGate, CZGate, ZGate, XGate, YGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes.optimization.commutation_analysis import _commute
from qiskit.quantum_info.operators import Clifford


def construct_dd(qc):
    dd = circuit_to_dagdependency(qc)
    return dd

    # gates = [ "s", "sdg", "]

def run_experiment(qc, is_cliff=True):
    # print("=====> RUNNING EXPERIMENT <=====")

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
    print(f"Transpile: #gates = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")
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
    print(f"InverseCancellation: #gates = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")
    if is_cliff:
        cliff2 = Clifford(qc2)
        # print(cliff2)
        ok = cliff == cliff2
        # print(f"{ok = }")
        assert ok
    # print("")

    # print("=== CommutativeInverseCancellation ===")
    start_time = time.time()
    pm2 = PassManager(CommutativeInverseCancellation(gates_to_cancel=gates_to_cancel))
    qc2 = pm2.run(qc)
    end_time = time.time()
    ops2 = qc2.count_ops()
    # print(f"ops2: {ops2}")
    # print(f"#gates = {sum([ops2[x] for x in ops2])}")
    # print(f"time = {end_time-start_time:.4f}")
    print(f"CommutativeInverseCancellation: #gates = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")
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
    print(f"CommutativeCancellation: #gates = {sum([ops2[x] for x in ops2])}, time = {end_time-start_time:.4f}")

    # if is_cliff:
    #     cliff2 = Clifford(qc2)
    #     print(cliff2)
    #     ok = cliff == cliff2
    #     print(f"{ok = }")
    #     assert ok
    # print("")

    print("")


def test_clifford(num_qubits, num_gates):
    gates = ["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"]
    for seed in range(10):
        qc = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed)
        run_experiment(qc)


def test_circ1():
    print(f"==========RUNNING EXPERIMENT 1")
    qc = QuantumCircuit(2)
    qc.z(0)
    qc.x(1)
    qc.cx(0, 1)
    qc.z(0)
    qc.x(1)
    print(qc)
    run_experiment(qc)



def test_circ2():
    print(f"========RUNNING EXPERIMENT 2")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    print(qc)
    run_experiment(qc)


if __name__ == "__main__":
    test_clifford(10, 5000)
    print("")

    # test_clifford(10, 5000)
    # test_clifford(50, 1000)
    # test_clifford(10, 1000)
    #test_circ1()
    # test_circ2()


