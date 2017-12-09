import argparse
import sys
import os
rootFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(rootFolder)

from extensible_gate_domain import _basic_gates_string_IBM, _basic_gates_string_IBM_advanced
from math_executor import math_execute
from circuit_builder import buildAST, buildCircuit
from circuit_layer_analyzer import gate_qubit_tuples_of_circuit_as_layers
import os
import math
import random
from mutator import replaceGate, replaceBits, replaceInstruction, buildRandomInstruction, mutate
from mutator import unaryOPs, binaryOPs, ternaryOPs, allOPs

from ibm_official import get_code_image, get_execution_result, get_image_and_result

# need to refactor the name "basis_gates"

# highlights:
# standalone:
#   visualize a circuit
#   math execution
#   support of measurement, partial view of the complete state
# check equivalence
# regression testing

_basic_gates_string = None

def makeFuzzArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basis_gates_option', type=int, default=2,
                           help='1: ibm q, 2: ibm q advanced')
    parser.add_argument('--qasm_files', type=str, default="./testcases/grover11.qasm", #
                       help='paths of the qasm files') # they are separated with

    parser.add_argument('--return_measured_state', type=int, default=1,
                   help='for math execution, 1 return measured state only, 0 return complete state')


    parser.add_argument('--fuzz_folder', type=str, default="./fuzz", #
                       help='folder for holding mutant files') # they are separated with

    parser.add_argument('--mode', type=int, default=2, #
                       help='1 single file 2 batch') # they are separated with

    parser.add_argument('--batch_count', type=int, default=20, #
                       help='if in batch mode, how many mutants to produce') # they are separated with


    parser.add_argument('--mutate_option', type=int, default=4, #
                       help='if in single mode, we support: 1 remove 2 insert 3 swap 4 replace gate 5 replace data 6 replace to a different instr') # they are separated with


    parser.add_argument('--ugate_enabled', type=int, default=1, #
                       help='u gate enabled? 0=no, 1 = yes') # they are separated with

    parser.add_argument('--device', type=str, default="ibmqx2", #
                       help='ibmqx2 or simulator') # only these two are possible

    #

    args = parser.parse_args()

    global _basic_gates_string
    if args.basis_gates_option == 1:
        _basic_gates_string = _basic_gates_string_IBM
    else:
        _basic_gates_string = _basic_gates_string_IBM_advanced


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








def fuzz_one_file(args, orig_file):
    orig_ast = buildAST(orig_file)
    global _basic_gates_string
    orig_circuit = buildCircuit(orig_ast, basis_gates=_basic_gates_string.split(","))
    orig_schedule_list_of_layers, global_qubit_inits, global_cbit_inits = gate_qubit_tuples_of_circuit_as_layers(orig_circuit)
    #orig_matrix, _, _ = math_execute(args, orig_circuit)

    # proposal:
    bitDomain = global_qubit_inits.keys()
    bitDomainSize = len(bitDomain)
    if not os.path.exists(args.fuzz_folder):
        raise SystemError("fuzz folder does not exist")

    ret = []
    if args.mode == 1: # single mode needs option
        newFileName = "single_mode_option_" + str(args.mutate_option)
        print "\n\nnew mutant written to: " + newFileName
        newFilePath = os.path.join(args.fuzz_folder, newFileName)
        mutate(orig_file, bitDomain, newFilePath, args.mutate_option)
        qasm_code = open(newFilePath).read()
        shots = 2**bitDomainSize if 2**bitDomainSize > 128 else 128
        image, result = get_image_and_result(qasm_code, ibmqx_shots=shots, backend=args.device)

        print image
        print result
        if isinstance(image,dict):
            image = None
            print "error occurs, transforms image to None"


        ret.append({'file': open(newFilePath).read(), 'image': image, 'result': result})
        return ret


        # return 1 string content 2 return result and status
    elif args.mode == 2: # batch mode needs count
        for batchid in range(args.batch_count):
            newFileName = "batch_mode_id_" + str(batchid)
            print "\n\nnew mutant written to: " + newFileName

            newFilePath = os.path.join(args.fuzz_folder, newFileName)
            mutate(orig_file, bitDomain, newFilePath, None) # None means random choice
            # return results and status
            qasm_code = open(newFilePath).read()
            shots = 2**bitDomainSize if 2**bitDomainSize > 128 else 128

            #ret.append({'file': open(newFilePath).read(), 'image': None, 'result': None})

            image, result = get_image_and_result(qasm_code, ibmqx_shots=shots, backend=args.device)
            print image
            print result
            if isinstance(image,dict):
                image = None
                print "error occurs, transforms image to None"

            ret.append({'file': open(newFilePath).read(), 'image': image, 'result': result})

        return ret
    else:
        raise SystemError("this mode value is not supported yet")


def qfuzzmain(args):
    if args.ugate_enabled:
        global unaryOPs, binaryOPs, allOPs
        unaryOPs.extend(['u1', 'u2', 'u3'])
        binaryOPs.extend(['cu1', 'cu3'])
        allOPs.extend(['u1', 'u2', 'u3', 'cu1', 'cu3'])

    qasm_paths_str = args.qasm_files
    result = parse_paths_linux_style(qasm_paths_str)

    ret = []

    for qasm_file in result:
        tmpRet = fuzz_one_file(args, qasm_file)
        ret.extend(tmpRet)

    return ret


if __name__ == '__main__':
    args = makeFuzzArgs()
    qfuzzmain(args)

