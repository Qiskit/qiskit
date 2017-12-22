import argparse

from extensible_gate_domain import _basic_gates_string_IBM, _basic_gates_string_IBM_advanced
from math_executor import math_execute
import os



from qiskit import qasm, unroll



def makeArgs():
    parser = argparse.ArgumentParser()
    currentFolder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--basis_gates_option', type=int, default=2,
                           help='1: ibm q, 2: ibm q advanced')

    parser.add_argument('--qasm_file', type=str, default=currentFolder + "/testcases/naive.qasm", #
                       help='paths of the qasm files') # they are separated with


    parser.add_argument('--return_measured_state', type=int, default=1,
                   help='1 return measured state, 0 return complete state')
    parser.add_argument('--report_after_every_step', type=int, default=1,
                   help='report the state after every step, otherwise, we report the final state only')



    args = parser.parse_args()
    print(vars(args))

    return args

# the string can be file1:file2:dir1:dir2
# if it is a dir, we will look for *.qasm inside the dir
def parse_paths_linux_style(qasm_paths_str):
    qasm_paths = qasm_paths_str.split(':')
    result = []
    for qasm_path in qasm_paths:
        if os.path.isdir(qasm_path):
            for root, directories, files in os.walk(qasm_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if filepath.endswith('.qasm') and filepath not in result:
                        result.append(filepath)  # Add it to the list.
        else:
            if os.path.isfile(os.path.abspath(qasm_path)) and qasm_path.endswith('.qasm') and qasm_path not in result:
                result.append(qasm_path)
    return result



def buildAST(qasm_file):
    if not qasm_file:
        print('"Not filename provided')
        return {"status": "Error", "result": "Not filename provided"}
    ast = qasm.Qasm(filename=qasm_file).parse()  # Node (AST)
    return ast


def buildCircuit(ast, basis_gates=None):
    unrolled_circuit = unroll.Unroller(ast=ast, backend=unroll.DAGBackend(basis_gates)) #CircuitBackend
    unrolled_circuit.execute()
    circuit_unrolled = unrolled_circuit.backend.circuit  # circuit DAG
    return circuit_unrolled








if __name__ == '__main__':
    args = makeArgs()
    if args.basis_gates_option == 1:
        _basic_gates_string = _basic_gates_string_IBM
    else:
        _basic_gates_string = _basic_gates_string_IBM_advanced

    qasm_file = args.qasm_file
    # result = parse_paths_linux_style(qasm_paths_str)
    # for qasm_file in result:
    print("\n\n\n\n > ANALYZING " + qasm_file)
    ast = buildAST(qasm_file)
    circuit = buildCircuit(ast, basis_gates=_basic_gates_string.split(","))
    math_execute(args, circuit)
