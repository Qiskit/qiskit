import argparse

import sys
import os
rootFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(rootFolder)


from equivalence_checker import check_equivalence
from extensible_gate_domain import _basic_gates_string_IBM, _basic_gates_string_IBM_advanced
from math_executor import math_execute
from circuit_builder import buildAST, buildCircuit
from visualizer import visualize, save_plot_only
from regression_test import regression_test
import os

# need to refactor the name "basis_gates"

# highlights:
# standalone:
#   visualize a circuit
#   math execution
#   support of measurement, partial view of the complete state
# check equivalence
# regression testing


def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basis_gates_option', type=int, default=2,
                           help='1: ibm q, 2: ibm q advanced')
    parser.add_argument('--mode', type=int, default=2,
                       help='1 check equivalence, 2 standalone (visualize and math execution) 3 regression testing')
    parser.add_argument('--qasm_files', type=str, default="/Users/liup/quantum/qiskit-sdk-py/qiskit/qanalyzer/testcases/grover11_opt.qasm", #
                       help='paths of the qasm files') # they are separated with

    #[section: visualize]
    parser.add_argument('--visualize', type=int, default=1,
                       help='do we do analysis over graph')
    parser.add_argument('--visualize_dot', type=int, default=0,
                   help='use dot ?')
    parser.add_argument('--visualize_circuit', type=int, default=0,
                   help='use quantum circuit ?')
    parser.add_argument('--visualize_circuit_layers', type=int, default=1,
                   help='do we do analysis over layers')
    #[section: math_execution]
    parser.add_argument('--math_execution', type=int, default=1,
                       help='shall we do math execution')
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



def qcheckermain(args):
    # important: the argument basis_gates control what gates are basic and can appear in the final assemblied circuit
    # the gates can be found in qiskit/extensions/standard/__init__.py
    if args.basis_gates_option == 1:
        _basic_gates_string = _basic_gates_string_IBM
    else:
        _basic_gates_string = _basic_gates_string_IBM_advanced

    if args.mode == 1:
        qasm_paths_str = args.qasm_files
        result = parse_paths_linux_style(qasm_paths_str)
        check_equivalence(args, result, _basic_gates_string)
    elif args.mode == 2:
        qasm_paths_str = args.qasm_files
        result = parse_paths_linux_style(qasm_paths_str)
        for qasm_file in result:
            print("\n\n\n\n > ANALYZING " + qasm_file)
            ast = buildAST(qasm_file)

            circuit = buildCircuit(ast, basis_gates=_basic_gates_string.split(",")) # as shown in IBM qunatum computing # you can add more
            save_plot_only(circuit)
            #import generate_svg_circuit
            #print generate_svg_circuit.SVGenWithCircuit(circuit).generate_svg()
            if args.visualize:
                visualize(args, circuit)
            if args.math_execution:
                math_execute(args, circuit)

    elif args.mode ==3:
        args.qasm_files = "../testcases"
        qasm_paths_str = args.qasm_files
        result = parse_paths_linux_style(qasm_paths_str)
        regression_test(args, result, _basic_gates_string)


    # QP_program = QuantumProgram()




if __name__ == '__main__':
    args = makeArgs()
    qcheckermain(args)

