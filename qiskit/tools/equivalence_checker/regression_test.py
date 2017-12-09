from circuit_builder import buildAST, buildCircuit
from math_executor import math_execute
from sympy import E

def regression_test(args,result, _basic_gates_string):
    for qasm_file in result:
        if 'grover_4bits.qasm' not in qasm_file: # switch to focus on one test case
            continue
        print("\n\n\n\n > TESTING " + qasm_file)
        ast = buildAST(qasm_file)
        circuit = buildCircuit(ast, basis_gates=_basic_gates_string.split(",")) # as shown in IBM qunatum computing # you can add more
        matrix, header = math_execute(args, circuit)
        if 'grover00.qasm' in qasm_file:
            assert matrix[0,0] == 1
            assert matrix[1,0] == 0
            assert matrix[2,0] == 0
            assert matrix[3,0] == 0
        elif 'grover01_name_convention.qasm' in qasm_file:
            assert matrix[0,0] == 0
            assert matrix[1,0] == 1
            assert matrix[2,0] == 0
            assert matrix[3,0] == 0
        elif 'grover10_name_convention.qasm' in qasm_file:
            assert matrix[0,0] == 0
            assert matrix[1,0] == 0
            assert matrix[2,0] == 1
            assert matrix[3,0] == 0
        elif 'grover11.qasm' in qasm_file:
            assert matrix[0,0] == 0
            assert matrix[1,0] == 0
            assert matrix[2,0] == 0
            assert matrix[3,0] == 1
        elif '2bitswap.qasm' in qasm_file:
            assert matrix[0,0] == 0
            assert matrix[1,0] == 1
            assert matrix[2,0] == 0
            assert matrix[3,0] == 0
        elif 'toffoli_without_swap.qasm' in qasm_file:
            assert matrix[0,0] == 0
            assert matrix[1,0] == 0
            assert matrix[2,0] == 0
            assert matrix[3,0] == 0
            assert matrix[4,0] == 0
            assert matrix[5,0] == 0
            assert matrix[6,0] == 0
            assert matrix[7,0] == 1
        elif 'deutschNis3.qasm' in qasm_file:
            assert matrix[0,0] == 0
            assert matrix[1,0] == E**0/2 # i.e., 1/2
            assert matrix[2,0] == 0
            assert matrix[3,0] == E**0/2
            assert matrix[4,0] == 0
            assert matrix[5,0] == E**0/2
            assert matrix[6,0] == 0
            assert matrix[7,0] == E**0/2
        elif 'grover_4bits.qasm' in qasm_file:
            for i in range(16):
                if i == 15:
                    assert matrix[i,0]*16 == 11
                else:
                    assert matrix[i,0]*16 == 3







