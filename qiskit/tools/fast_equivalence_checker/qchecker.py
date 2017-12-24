import argparse

from equivalence_checker import check_equivalence
from extensible_gate_domain import _basic_gates_string_IBM, _basic_gates_string_IBM_advanced
from math_executor import math_execute
import os

from qiskit import qasm, unroll







def makeArgs():
    parser = argparse.ArgumentParser()
    currentFolder = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--basis_gates_option', type=int, default=2,
                           help='1: ibm q, 2: ibm q advanced')

    parser.add_argument('--file1', type=str, default=currentFolder+"/testcases/q3sat_small.qasm", #
                       help='file1 for comparison') # they are separated with

    parser.add_argument('--file2', type=str, default=currentFolder+"/testcases/q3sat_small_equivalent.qasm", #
                       help='file2 for comparison') # they are separated with
    parser.add_argument('--return_measured_state', type=int, default=0,
                   help='1 return measured state, 0 return complete state')
    parser.add_argument('--report_after_every_step', type=int, default=0,
                   help='report the state after every step, otherwise, we report the final state only')


    args = parser.parse_args()

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
            if os.path.isfile(qasm_path) and qasm_path.endswith('.qasm') and qasm_path not in result:
                result.append(qasm_path)
    return result



def qcheckermain(args):
    # important: the argument basis_gates control what gates are basic and can appear in the final assemblied circuit
    # the gates can be found in qiskit/extensions/standard/__init__.py
    if args.basis_gates_option == 1:
        _basic_gates_string = _basic_gates_string_IBM
    else:
        _basic_gates_string = _basic_gates_string_IBM_advanced

    check_equivalence(args, args.file1, args.file2, _basic_gates_string)





if __name__ == '__main__':
    args = makeArgs()
    qcheckermain(args)

