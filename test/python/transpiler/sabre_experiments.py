import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import TransformationPass, CouplingMap, PassManager, Layout
from qiskit.transpiler.passes import SabreSwap
from qiskit.circuit.library import Permutation
from qiskit.converters import circuit_to_dag, dag_to_circuit, dag_to_dagdependency


def create_circ1(n):
    qc = QuantumCircuit(n)
    for i in range(1, n):
        qc.cx(0, i)
    return qc


def create_circ2(n):
    qc = QuantumCircuit(n)
    for i in range(n-1, 0, -1):
        qc.cx(0, i)
    return qc


def create_circ3(n):
    qc = QuantumCircuit(n)
    for i in range(1, n):
        qc.cx(0, i)
    qc.h(0)
    for i in range(n - 1, 0, -1):
        qc.cx(0, i)
    return qc


def create_circ4():
    qc = QuantumCircuit.from_qasm_file("20QBT_45CYC_.0D1_.1D2_3.qasm")
    return qc


def run_sabre(qc, heuristic="decay", do_commutative_analysis=True, check_equivalence=True):
    """Run SABRE-based routing pass."""
    dag = circuit_to_dag(qc)
    n = qc.num_qubits
    coupling_list = []
    for i in range(0, n-1):
        coupling_list.append([i, i+1])
        coupling_list.append([i+1, i])
    coupling_map = CouplingMap(coupling_list)
    sabre_swap = SabreSwap(coupling_map=coupling_map, heuristic=heuristic, seed=0, do_commutative_analysis=do_commutative_analysis)

    tdag = sabre_swap.run(dag)
    tqc = dag_to_circuit(tdag)

    if check_equivalence:
        transpiled_layout = sabre_swap.property_set['final_layout']
        check_routing(qc, tqc, transpiled_layout)

    return tqc


def check_routing(circ, transpiled_circ, transpiled_layout):
    """Checking equivalence of original and routed circuits."""
    nq = circ.num_qubits

    if nq >= 10:
        print(f"Too Large: {nq = }")
        return

    perm_pattern = [transpiled_layout._v2p[v] for v in circ.qubits]
    transpiled_circ_tmp = transpiled_circ.copy()
    transpiled_circ_tmp.append(Permutation(nq, perm_pattern), range(nq))

    op1 = Operator(circ)
    op2 = Operator(transpiled_circ_tmp)

    eq = np.all(op1 == op2)
    print(f"Equivalent: {eq}")
    assert eq


def print_circuit_stats(header, qc, verbosity=1):
    """print circuit info"""
    print(f"{header}: #ops = {qc.size()}, depth = {qc.depth()}; ops = {qc.count_ops()}")
    if verbosity>=2:
        print(qc)


def quick_experiments():
    "A few quick experiments"
    circs = [create_circ1(8), create_circ2(8), create_circ3(8)]

    for qc in circs:
        print("=========================")
        print_circuit_stats("Original", qc, 2)
        tqc = run_sabre(qc, heuristic="basic", do_commutative_analysis=False, check_equivalence=True)
        print_circuit_stats("Routed [-comm, basic]", tqc, 1)
        tqc = run_sabre(qc, heuristic="basic", do_commutative_analysis=True, check_equivalence=True)
        print_circuit_stats("Routed [+comm, basic]", tqc, 1)
        tqc = run_sabre(qc, heuristic="lookahead", do_commutative_analysis=False, check_equivalence=True)
        print_circuit_stats("Routed [-comm, ahead]", tqc, 1)
        tqc = run_sabre(qc, heuristic="lookahead", do_commutative_analysis=True, check_equivalence=True)
        print_circuit_stats("Routed [+comm, ahead]", tqc, 1)
        tqc = run_sabre(qc, heuristic="decay", do_commutative_analysis=False, check_equivalence=True)
        print_circuit_stats("Routed [-comm, decay]", tqc, 1)
        tqc = run_sabre(qc, heuristic="decay", do_commutative_analysis=True, check_equivalence=True)
        print_circuit_stats("Routed [+comm, decay]", tqc, 1)


def bench():
    """Run some kind of benchmarking on some of the red-queen's routing problems"""

    from os import listdir
    from os.path import isfile, join
    # qasm_dir = "../red-queen/red_queen/games/applications/qasm"
    qasm_dir = "../red-queen/red_queen/games/mapping/benchmarks/misc"
    # qasm_dir = "../red-queen/red_queen/games/mapping/benchmarks/queko/BIGD"
    # qasm_dir = "../MQT"

    res_names = []
    res_nq = []     # number of qubits
    res_ng1 = []    # number of single-qubit gates
    res_ng2 = []    # number of two-qubit gates
    res_swaps1 = []
    res_swaps2 = []
    res_swaps3 = []
    res_swaps4 = []
    res_swaps5 = []
    res_swaps6 = []

    # depth is not computed correctly (as SWAPS are counted as 1, not as 3)
    res_depth1 = []
    res_depth2 = []
    res_depth3 = []
    res_depth4 = []
    res_depth5 = []
    res_depth6 = []

    print(listdir(qasm_dir))
    qasm_files = [f for f in listdir(qasm_dir) if isfile(join(qasm_dir, f)) and f.endswith(".qasm")]

    cur_test = 0
    max_tests = 1000000

    #for f in ["9symml_195.qasm"]:

    for f in qasm_files:
        cur_test+=1
        if cur_test > max_tests:
            break

        qc = QuantumCircuit.from_qasm_file(join(qasm_dir, f))
        # print(qc)
        print(f"Running {f}, #qubits = {qc.num_qubits}, #gates = {len(qc.data)}")
        if len(qc.data) >= 10000:
            print("SKIPPING, too large")
            continue

        tqc1 = run_sabre(qc, heuristic="basic", do_commutative_analysis=False, check_equivalence=True)
        print_circuit_stats("=> B -C", tqc1)
        tqc2 = run_sabre(qc, heuristic="basic", do_commutative_analysis=True, check_equivalence=True)
        print_circuit_stats("=> B +C", tqc2)
        tqc3 = run_sabre(qc, heuristic="lookahead", do_commutative_analysis=False, check_equivalence=True)
        print_circuit_stats("=> D -L", tqc3)
        tqc4 = run_sabre(qc, heuristic="lookahead", do_commutative_analysis=True, check_equivalence=True)
        print_circuit_stats("=> D +L", tqc4)
        tqc5 = run_sabre(qc, heuristic="decay", do_commutative_analysis=False, check_equivalence=True)
        print_circuit_stats("=> D -C", tqc5)
        tqc6 = run_sabre(qc, heuristic="decay", do_commutative_analysis=True, check_equivalence=True)
        print_circuit_stats("=> D +C", tqc6)

        print("===")

        res_names.append(f)
        res_nq.append(qc.num_qubits)
        res_ng1.append(qc.size(filter_function=lambda x: not x.operation._directive and len(x.qubits) == 1))
        res_ng2.append(qc.size(filter_function=lambda x: not x.operation._directive and len(x.qubits) == 2))
        res_swaps1.append(tqc1.count_ops().get("swap", 0))
        res_swaps2.append(tqc2.count_ops().get("swap", 0))
        res_swaps3.append(tqc3.count_ops().get("swap", 0))
        res_swaps4.append(tqc4.count_ops().get("swap", 0))
        res_swaps5.append(tqc5.count_ops().get("swap", 0))
        res_swaps6.append(tqc6.count_ops().get("swap", 0))

        res_depth1.append(tqc1.depth())
        res_depth2.append(tqc2.depth())
        res_depth3.append(tqc3.depth())
        res_depth4.append(tqc4.depth())
        res_depth5.append(tqc5.depth())
        res_depth6.append(tqc6.depth())

    print("NAME                      NQ    1Q      2Q       |     B -C    B +C    L -C    L +C    D -C    D +C")
    for i in range(len(res_names)):
        print(f"{res_names[i]:26}{res_nq[i]:6}{res_ng1[i]:8}{res_ng2[i]:8} | "
              f"{res_swaps1[i]:8}{res_swaps2[i]:8}{res_swaps3[i]:8}{res_swaps4[i]:8}{res_swaps5[i]:8}{res_swaps6[i]:8}")


if __name__ == "__main__":
    # quick_experiments()
    bench()
