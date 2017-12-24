import itertools
from circuit_builder import buildAST, buildCircuit
from math_executor import math_execute
from sympy import Matrix, solve_linear_system, solve, Eq, I, E, pretty, log
from sympy.abc import x, y
from sympy.physics.quantum.represent import represent

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
        maxlength = max([len(str(qm1[i,0])) for i in range(N1)])
        formatstr = '|{:'+ str(maxlength)+'s}|'

        for i in range(N1):
            # print(formatstr.format(str(qm1[i,0])) + formatstr.format(str(qm2[i,0])))
            eqtmp = Eq(E**(I*x)*qm1[i,0], qm2[i,0])
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

def check_equivalence(args, qasm_file1, qasm_file2, _basic_gates_string):
    #_basic_gates = _basic_gates_string.split(",")
    # for qasm_file1, qasm_file2 in itertools.combinations(qasm_files, 2):

    ast1 = buildAST(qasm_file1)
    circuit1 = buildCircuit(ast1, basis_gates=_basic_gates_string.split(",")) # as shown in IBM qunatum computing # you can add more
    represent1, label1, stringrep1 = math_execute(args, circuit1)

    ast2 = buildAST(qasm_file2)
    circuit2 = buildCircuit(ast2, basis_gates=_basic_gates_string.split(",")) # as shown in IBM qunatum computing # you can add more
    represent2, label2, _ = math_execute(args, circuit2)

    if label1 == label2:
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
        print("We cannot check the equivalence because the lists of qubits do not precisely match! ")


