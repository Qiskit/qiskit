import argparse
import os

from qiskit import qasm, unroll



import itertools
from sympy import Matrix, solve_linear_system, solve, Eq, I, E, pretty, log
from sympy.abc import x, y
from sympy.physics.quantum.represent import represent
import os
from qiskit import QuantumCircuit, QuantumProgram



def makeArgs():
    parser = argparse.ArgumentParser()
    currentFolder = os.path.dirname(os.path.realpath(__file__))


    parser.add_argument('--file1', type=str, default=currentFolder+"/testcases/q3sat_small.qasm", #
                       help='file1 for comparison') # they are separated with

    parser.add_argument('--file2', type=str, default=currentFolder+"/testcases/q3sat_small_equivalent.qasm", #
                       help='file2 for comparison') # they are separated with



    args = parser.parse_args()

    return args





# previously, we use qubit as the argument and we needed represent(qubit) to get the matrix
# now we directly feed the matrix representations as the arguments
def check_equivalent_between_two_results(qm1, qm2):
    #Eq(E**(I*x)
    eqlist = []
    # qm1 = represent(q1) # Nx1 2-d matrix for representing a N-entry vector
    N1 = qm1.shape[0]
    # qm2 = represent(q2) # Nx1 2-d matrix for representing a N-entry vector
    N2 = qm2.shape[0]

    print("****************************")
    if N1 != N2:
        raise Exception('wrong') # don't, if you catch, likely to hide bugs.
    else:
        for i in range(N1):
            eqtmp = Eq(E**(I*x)*qm1[i], qm2[i])
            if eqtmp == True:
                continue # safely ignore the constraint that is always true
            elif eqtmp == False:
                return False, None
            else:
                eqlist.append(eqtmp)




    result = solve(eqlist,  [x])
    print("****************************")


    #result = solve([Eq(1*x, I), Eq(1*y, I), Eq(x*x + y*y, 1)],  [x, y])
    if len(result) != 0:
        return True, result[x]
    else:
        return False, None


def compute_amplitude_vector(qasm_file):
    Q_program = QuantumProgram()
    myqasm = Q_program.load_qasm_file(qasm_file, "my_example")
    circuits = ['my_example'] #, 'superposition'
    backend = 'local_sympy_qasm_simulator' # the device to run on
    result = Q_program.execute(circuits, backend=backend, shots=1, timeout=300)
    return result.get_data('my_example')['quantum_state']

def check_equivalence(qasm_file1, qasm_file2):
    represent1 = compute_amplitude_vector(qasm_file1)
    represent2 = compute_amplitude_vector(qasm_file2)



    if len(represent1) == len(represent2):
        isEquivalent, solution = check_equivalent_between_two_results(represent1, represent2)
        if isEquivalent:
            print("the final amplitude vectors (namely, V1 and V2) of the following two are equivalent:")
            print(" "+qasm_file1)
            print(" "+qasm_file2)
            print("because V1 * E^(I*x) = V2 has a solution, where x is " + str(solution))
        else:
            print("the final amplitude vectors (namely, V1 and V2) of the following two are NOT equivalent:")
            print(" "+qasm_file1)
            print(" "+qasm_file2)
            print("because V1 * E^(I*x) = V2 does not have a solution!")
    else:
        print("We cannot check the equivalence because the size of the final amplitude vector does not match! ")





# the string can be file1:file2:dir1:dir2
# if it is a dir, we will look for *.qasm inside the dir
# def parse_paths_linux_style(qasm_paths_str):
#     qasm_paths = qasm_paths_str.split(':')
#     result = []
#     for qasm_path in qasm_paths:
#         if os.path.isdir(qasm_path):
#             for root, directories, files in os.walk(qasm_path):
#                 for filename in files:
#                     filepath = os.path.join(root, filename)
#                     if filepath.endswith('.qasm') and filepath not in result:
#                         result.append(filepath)  # Add it to the list.
#         else:
#             if os.path.isfile(qasm_path) and qasm_path.endswith('.qasm') and qasm_path not in result:
#                 result.append(qasm_path)
#     return result



def qcheckermain(args):
    check_equivalence(args.file1, args.file2)





if __name__ == '__main__':
    args = makeArgs()
    qcheckermain(args)

