from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.library import BridgeGate
from qiskit.circuit.library.standard_gates import CXGate, SwapGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization import SwapCXSwapToBridge, CollectLinearFunctions
from qiskit.transpiler.passes import *
from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit.circuit import EquivalenceLibrary
from qiskit.transpiler.synthesis import cnot_synth
from qiskit.quantum_info import Operator
import logging

def test_bridge():
    qc = QuantumCircuit(6)
    qc.swap(1, 2)
    qc.swap(2, 3)
    qc.cx(3, 4)
    qc.swap(2, 3)
    qc.swap(1, 2)
    print("Original Circuit:")
    print(qc)

    pm = PassManager(SwapCXSwapToBridge())

    print("Simplified Circuit:")
    circuit = pm.run(qc)
    print(circuit)

def create_circuit0():
    qc = QuantumCircuit(3)
    qc.swap(0, 1)
    qc.swap(1, 2)
    return qc

def create_circuit1():
    qc = QuantumCircuit(6)
    qc.swap(1, 2)
    qc.swap(2, 3)
    #qc.h(2)
    qc.cx(3, 4)
    qc.swap(2, 3)
    qc.swap(1, 2)
    return qc

def create_circuit2():
    qc = QuantumCircuit(9)
    qc.h(2)
    qc.cx(1, 3)
    qc.swap(3, 5)
    qc.cx(3, 4)
    qc.cx(1, 4)
    qc.h(1)
    qc.cx(7, 8)
    qc.cx(8, 7)
    qc.cx(7, 8)
    return qc

def create_circuit5():
    qc = QuantumCircuit(9)
    qc.h(2)
    qc.swap(1, 3)
    qc.swap(1, 3)
    qc.h(3)
    return qc

def create_circuit4():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    #qc.h(2)
    qc.h(3)
    qc.cx(1, 2)
    qc.h(3)
    return qc

def create_circuit3():
    qc = QuantumCircuit(9)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.h(7)
    qc.cx(3, 4)
    qc.cx(4, 5)
    return qc

def test_linear():

    qc = create_circuit1()
    print("Original Circuit:")
    print(qc)

    logging.basicConfig(level='INFO')
    #logging.getLogger('qiskit.transpiler').setLevel('DEBUG')

    pm = PassManager(CollectLinearFunctions())

    print("Simplified Circuit:")
    circuit = pm.run(qc)
    print(circuit)

    for inst, qargs, cargs in circuit.data:
        print(f"inst  = {inst}")
        print(f"qargs = {qargs}")
        print(f"cargs = {cargs}")
        if isinstance(inst, Instruction):
            print(f"I AM AN INSTRUCTION")
            print(f"{inst.definition}")
            print(f"params = {inst.params}")

    print("Transpiling")
    pm2 = PassManager(Unroll3qOrMore())
    circuit2 = pm2.run(circuit)
    #circuit2 = transpile(circuit, optimization_level=3)
    print(circuit2)




import numpy as np
def test_cnot_synth():
    qc = create_circuit1()
    print(qc)

    nq = qc.num_qubits
    arr = np.eye(nq, nq, dtype=bool)
    print(f"of type {type(arr)}")
    print(arr)

    bit_indices = {bit: idx for idx, bit in enumerate(qc.qubits)}
    print(f"bit_indices = {bit_indices}")

    for inst, qargs, cargs in qc.data:
        print(f"inst  = {inst}")
        print(f"qargs = {qargs}")
        print(f"cargs = {cargs}")
        if inst.name == "cx":
            print(f"--> This is CX")
            c = bit_indices[qargs[0]]
            t = bit_indices[qargs[1]]
            print(f"c = {c}")
            print(f"t = {t}")
            print(f"c-row: {arr[c, :]}")
            print(f"t-row: {arr[t, :]}")
            arr[t, :] = (arr[t, :]) ^ (arr[c, :])
            print("AFTER:")
            print(arr)

        elif inst.name == "swap":
            print(f"--> This is SWAP")
            c = bit_indices[qargs[0]]
            t = bit_indices[qargs[1]]
            print(f"c = {c}")
            print(f"t = {t}")
            print(f"c-row: {arr[c, :]}")
            print(f"t-row: {arr[t, :]}")
            arr[[c, t]] = arr[[t, c]]
            print("AFTER:")
            print(arr)

        else:
            print(f"---> NOT SUPPORTED")
        print(f"")

    circ = cnot_synth(arr)
    print("AFTER TRANSPILE:")
    print(circ)
    print(Operator(circ))
    print(Operator(qc))
    eq = Operator(qc) == Operator(circ)
    print(f"same = {eq}")

from qiskit.converters import *

def analyze_dag():
    qc = create_circuit5()
    print(qc)

    dag = circuit_to_dag(qc)

    for node in dag.topological_op_nodes():
        print(f"NODE: {node}")
        print(f"--node name: {node.name}")
        print(f"--node op:   {node.op}")
        print(f"--node qargs: {node.qargs}")
        print(f"--node cargs: {node.cargs}")
        print(f"--node condition: {node.condition}")
        edges = dag.edges(node)
        pred = dag.predecessors(node)
        succ = dag.successors(node)
        print(f"--node edges: {edges}")
        for x in edges:
            print(f"----{x}")
        print(f"--node predecessors: {pred}")
        for x in pred:
            print(f"----{x}")
        print(f"--node successors  : {succ}")
        for x in succ:
            print(f"----{x}")
        print(f"")



def test_collect_runs():
    qc = create_circuit5()
    print(qc)

    dag = circuit_to_dag(qc)
    runs = dag.collect_runs("swap")
    print(f"Have {len(runs)} runs:")
    print(runs)

    for node_tuple in runs:
        print("NODE")
        for node in node_tuple:
            print(f"--node name: {node.name}")
            print(f"--node op:   {node.op}")
            print(f"--node qargs: {node.qargs}")
            print(f"--node cargs: {node.cargs}")
            print("")


def test_cx_cancel():
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.h(2)
    qc.h(3)
    qc.cx(1, 2)
    qc.h(3)
    print(qc)

    pm2 = PassManager(CXCancellation())
    circuit2 = pm2.run(qc)
    print(circuit2)

def test_qc():
    qc1 = QuantumCircuit(4)
    qc1.h(2)
    qc1.cx(2,3)

    qc2 = QuantumCircuit(6)
    qc2.cx(0, 1)
    print(f"HERE")
    qc2.append(qc1.to_gate(), [0, 1, 2, 3])
    print(qc2)


def test_qc():
    qc1 = QuantumCircuit(3)
    qc1.ccx(0, 1, 2)
    qc2 = QuantumCircuit(5)
    qc2.h(0)
    qc2.append(qc1, [0, 2, 4])
    print(qc2)
    qc3 = transpile(qc2, optimization_level=3)
    print(qc3)


def test_permutation():
    from qiskit.circuit.library.generalized_gates.permutation import _get_ordered_swap
    from qiskit.circuit.library.generalized_gates import LinearFunction
    from qiskit.quantum_info.operators import Clifford
    from qiskit.quantum_info.synthesis.clifford_decompose import decompose_clifford

    nq = 3
    permutation = np.random.permutation(nq)
    #permutation = [1, 2, 0]
    print(permutation)
    circuit = QuantumCircuit(nq, name="MyRandomPermutation")
    for i, j in _get_ordered_swap(permutation):
        circuit.swap(i, j)
    print(circuit)
    ops = circuit.count_ops()
    # print(f"ops = {ops}")
    num_swaps = ops["swap"] if "swap" in ops.keys() else 0
    print(f"PERM: Has {num_swaps} swaps; total cx cost = {3 * num_swaps}")
    linear = LinearFunction(nq, circuit)
    unroll_pass = PassManager(Unroll3qOrMore())
    circuit2 = unroll_pass.run(circuit)
    #for instr, _, _ in circuit2._data:
    #    print(f"inst = {instr}")
    ops = circuit2.count_ops()
    #print(f"ops2 = {ops}")
    num_swaps = ops["swap"] if "swap" in ops.keys() else 0
    num_cxs = ops["cx"] if "cx" in ops.keys() else 0
    print(f"PMH : Has {num_swaps} swaps and {num_cxs} cxs; total cx cost = {3 * num_swaps + num_cxs}")
    print(circuit2)
    if nq <= 6:
        oper1 = Operator(circuit)
        oper2 = Operator(circuit2)
        eq = oper1 == oper2
        print(f"Are we equal: {eq}")
    cliff = Clifford(circuit)
    print(cliff)
    circuit3 = decompose_clifford(cliff)
    print(circuit3)
    ops = circuit3.count_ops()
    num_swaps = ops["swap"] if "swap" in ops.keys() else 0
    num_cxs = ops["cx"] if "cx" in ops.keys() else 0
    print(f"CLF : Has {num_swaps} swaps and {num_cxs} cxs; total cx cost = {3 * num_swaps + num_cxs}")



if __name__ == "__main__":
    # test_bridge()
    #test_linear()
    test_permutation()


    #test_cx_cancel()
    #test_collect_runs()
    # TUNN ME ON
    # test_cnot_synth()

    # test_qc()
    #analyze_dag()
    #test_qc()