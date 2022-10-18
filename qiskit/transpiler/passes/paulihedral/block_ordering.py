# ......
#
#
#
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.library.pauli_evolution import PauliEvolutionKernel,PauliEvolutionKernels
from qiskit.opflow import I, Z, X,Y
#from PauliEvolutionKernel_bank import *
def list_to_PauliEvolustionGate(input:list)->PauliEvolutionGate:
    operator=0
    for i in range(len(input)-1):
        #i[0] IIIX i[1]0.75
        if input[i][0][0] == 'I':
            pauli_string=I
        if input[i][0][0] == 'X':
            pauli_string=X
        if input[i][0][0] == 'Z':
            pauli_string=Z
        if input[i][0][0] == 'Y':
            pauli_string=Y

        for j in input[i][0][1:]:
            if j == 'I':
                pauli_string = pauli_string ^ I
            elif j == 'X':
                pauli_string = pauli_string ^ X
            elif j == 'Z':
                pauli_string = pauli_string ^ Z
            elif j == 'Y':
                pauli_string = pauli_string ^ Y
        pauli_string=input[i][1]*pauli_string
        operator+=pauli_string
        #print(operator)
        pauli_string=''
    return PauliEvolutionGate(operator,time=input[-1])

def list_to_PauliEvolutionKernal(input:list)->PauliEvolutionKernel:
    result=PauliEvolutionKernel([])
    for i in input:
        result.kernel.append(list_to_PauliEvolustionGate(i))
    return result
    
def list_to_PauliEvolutionKernels(input:list)->PauliEvolutionKernels:
    result=PauliEvolutionKernels([])
    for i in input:
        result.kernels.append(list_to_PauliEvolutionKernal(i))
    return result

r"""

def lex_ordering(pauliIRprogram):####
    for i, block in enumerate(pauliIRprogram):
        param = block[-1]
        pauliIRprogram[i] = sorted(block[0:-1], key=str_lex_key)
        pauliIRprogram[i].append(param)
    pauliIRprogram = sorted(pauliIRprogram, key=block_lex_key)
    return pauliIRprogram

def gco_ordering(pauliIRprogram, *args):####
    pauliIRprogram = lex_ordering(pauliIRprogram)
    pauli_layers = [0] * len(pauliIRprogram)
    for i in range(len(pauliIRprogram)):
        pauli_layers[i] = [pauliIRprogram[i]]

    #print(pauli_layers)
    return pauli_layers

"""



def block_len(pauli_block: PauliEvolutionGate) -> int:
    '''example block: [XYX, p1], [XZZ, p2], ..., [ZZZ, p5], p]'''
    qubit_num = len(pauli_block.operator.paulis[0])
    length = 0
    for i in range(qubit_num):
        for pauli_str in pauli_block.operator.paulis:
            if str(pauli_str)[i] != 'I':
                length += 1
                break
    return length




def str_lex_key(weighted_pauli_str):
    value = 0
    '''q0 corresponds to the right-most pauli op'''
    weighted_pauli_str=weighted_pauli_str[0]
    for op in str(weighted_pauli_str):
        value *= 4
        if op == 'I':
            value += 0
        elif op == 'X':
            value += 1
        elif op == 'Y':
            value += 2
        elif op == 'Z':
            value += 3
    return -value







def block_lex_key(pauli_block: PauliEvolutionGate):
    value = 0
    '''q0 corresponds to the right-most pauli op'''
    for op in str(pauli_block.operator.paulis[0]):
        value *= 4
        if op == 'I':
            value += 0
        elif op == 'X':
            value += 1
        elif op == 'Y':
            value += 2
        elif op == 'Z':
            value += 3
    return -value




def block_latency(pauli_block: PauliEvolutionGate, qubit_num: int) -> int:
    latency = 0
    for pauli_str in pauli_block.operator.paulis:
        latency += max(2 * (qubit_num - str(pauli_str).count('I')) - 1, 0)
        if qubit_num - str(pauli_str).count('I') - str(pauli_str).count('Z') > 0:
            latency += 1  # or 2
    # print(latency)
    return latency




def block_qubit_occupation(pauli_block: PauliEvolutionGate, qubit_num: int) -> list:
    qubit_occupied = [0] * qubit_num
    for i in range(qubit_num):
        for pauli_str in pauli_block.operator.paulis:
            if str(pauli_str)[i] != 'I':
                qubit_occupied[i] = 1
            break
    #print("146block_qubit_occupation\n",qubit_occupied)
    return qubit_occupied



def qubit_occupation_template(qubit_occupied, qubit_num, latency):
    template = []
    start_index = 0
    end_index = 0

    '''0 -> qubit not occupied (I), 1 -> qubit occupied (X,Y,Z)'''
    while start_index < qubit_num:
        while end_index < qubit_num and qubit_occupied[end_index] == 0:
            end_index += 1
        if (end_index - start_index) >= 2:
            window = [1] * start_index + [0] * (end_index - start_index) + [1] * (qubit_num - end_index)
            template.append([window, latency])
        while end_index < qubit_num and qubit_occupied[end_index] != 0:
            end_index += 1
        start_index = end_index
    #print("166qubit_occupation_template\n",template)
    return template


def qubit_occupation_template(qubit_occupied: list, qubit_num: int, latency: int):
    template = []
    start_index = 0
    end_index = 0

    '''0 -> qubit not occupied (I), 1 -> qubit occupied (X,Y,Z)'''
    while start_index < qubit_num:
        while end_index < qubit_num and qubit_occupied[end_index] == 0:
            end_index += 1
        if (end_index - start_index) >= 2:
            window = [1] * start_index + [0] * (end_index - start_index) + [1] * (qubit_num - end_index)
            template.append([window, latency])
        while end_index < qubit_num and qubit_occupied[end_index] != 0:
            end_index += 1
        start_index = end_index
    #print("170\n",template)
    return template



def occupation_embedding_check(qubit_occupied_x: list, qubit_occupied_y: list):
    for i, occupied_x in enumerate(qubit_occupied_x):
        if occupied_x == 1 and qubit_occupied_y[i] == 1:
            return False
    return True



def qubit_occupation_combine(qubit_occupied_x: list, qubit_occupied_y: list) -> list:
    combined_occupation = [0] * len(qubit_occupied_x)
    for i, occupied_x in enumerate(qubit_occupied_x):
        if occupied_x == 1 or qubit_occupied_y[i] == 1:
            combined_occupation[i] = 1
    return combined_occupation


# def generate_templates1(ps, nq, latency, lt=2):
#     l = latency
#     idx0 = 0
#     idx1 = 0
#     t = []
#     while idx0 < nq:
#         while idx1 < len(ps) and ps[idx1] == "I":
#             idx1 += 1
#         if (idx1 - idx0) >= 2:
#             ts = (idx0-0)*'I'+(idx1-idx0)*'X'+(nq-idx1)*'I'
#             t.append([ts, l])
#         while idx1 < len(ps) and ps[idx1] != "I":
#             idx1 += 1
#         idx0 = idx1
#     return t




def lex_ordering(pauliIRprogram: PauliEvolutionKernel):  ####
    '''
    pauliIRprogram = [[['IIIX', 0.75], ['IIZX', -0.25], ['IZIX', -0.25], ['IZZX', -0.25], theta],
                  [['IIXI', 0.75], ['IIXZ', -0.25], ['IZXI', -0.25], ['IZXZ', -0.25], theta],
                  [['IXII', 0.375], ['IXIZ', -0.125], ['IXZI', -0.125], ['IXZZ', -0.125],
                   ['ZXII', -0.375], ['ZXIZ', 0.125], ['ZXZI', 0.125], ['ZXZZ', 0.125],
                   theta],
                  [['XIII', 0.5], ['XZII', -0.5], theta]]
    '''
    for ind, block in enumerate(pauliIRprogram.kernel):
        pauli_block = []
        for i in range(len(block.operator.paulis)):
            pauli_block.append([str(block.operator.paulis[i]), block.operator.coeffs[i].real])
        pauli_block = sorted(pauli_block, key=str_lex_key)
        pauli_block.append(block.time)
        pauliIRprogram.kernel[ind] = list_to_PauliEvolustionGate(pauli_block)
    # for i in range(pauliIRprogram)
    pauliIRprogram.kernel = sorted(pauliIRprogram.kernel, key=block_lex_key)
    return pauliIRprogram



def gco_ordering(pauliIRprogram: PauliEvolutionKernel, *args) -> PauliEvolutionKernels:  ####
    pauliIRprogram = lex_ordering(pauliIRprogram)
    result = PauliEvolutionKernels([])
    for i in range(len(pauliIRprogram.kernel)):
        a = PauliEvolutionKernel([])
        a.kernel.append(pauliIRprogram.kernel[i])
        result.kernels.append(
            a
        )
    return result



def do_ordering(pauliIRprogram: PauliEvolutionKernel, max_iteration=20) -> PauliEvolutionKernels:
    '''
    pauliIRprogram = [[['IIIX', 0.75], ['IIZX', -0.25], ['IZIX', -0.25], ['IZZX', -0.25], theta],
                  [['IIXI', 0.75], ['IIXZ', -0.25], ['IZXI', -0.25], ['IZXZ', -0.25], theta],
                  [['IXII', 0.375], ['IXIZ', -0.125], ['IXZI', -0.125], ['IXZZ', -0.125],
                   ['ZXII', -0.375], ['ZXIZ', 0.125], ['ZXZI', 0.125], ['ZXZZ', 0.125],
                   theta],
                  [['XIII', 0.5], ['XZII', -0.5], theta]]
    '''
    qubit_num = len(pauliIRprogram.kernel[0].operator.paulis[0])

    if qubit_num >= 4:
        large_block_threshold = qubit_num / 2
    else:
        large_block_threshold = 2

    large_block_list = PauliEvolutionKernel([])
    small_block_list = PauliEvolutionKernel([])

    for block in pauliIRprogram.kernel:  # gate
        if block_len(block) > large_block_threshold:
            large_block_list.kernel.append(block)
        else:
            small_block_list.kernel.append(block)

    large_block_list = lex_ordering(large_block_list)  # kernel
    small_block_list = lex_ordering(small_block_list)


    #print("large_block: ", len(large_block_list.kernel))
    #print("small_block: ", len(small_block_list.kernel))

    block_num = len(pauliIRprogram.kernel)

    pauli_layers = PauliEvolutionKernels([])

    while block_num > 0:
        if len(large_block_list.kernel) > 0:
            current_block = large_block_list.kernel[0]
            large_block_list = PauliEvolutionKernel(large_block_list.kernel[1:])
        elif len(small_block_list.kernel) > 0:
            current_block = small_block_list.kernel[0]
            small_block_list = PauliEvolutionKernel(small_block_list.kernel[1:])

        current_layer = PauliEvolutionKernel([current_block])  # kernel

        block_num -= 1

        latency = block_latency(current_block, qubit_num)
        layer_qubit_occupation = block_qubit_occupation(current_block, qubit_num)

        current_template = qubit_occupation_template(layer_qubit_occupation, qubit_num, latency)
        # print("285", current_template, qubit_num, latency)
        iteration_count = 0
        while len(current_template) > 0 and iteration_count < max_iteration:
            iteration_count += 1
            qubit_occupation_list = []
            for template in current_template:  # list list
                small_block_not_selected_list = PauliEvolutionKernel([])
                for small_block in small_block_list.kernel:
                    small_latency = block_latency(small_block, qubit_num)
                    small_occupation = block_qubit_occupation(small_block, qubit_num)
                    if small_latency <= template[1] and occupation_embedding_check(template[0], small_occupation):
                        current_layer.kernel.append(small_block)
                        qubit_occupation_list.append(small_occupation)
                        template[1] -= small_latency
                        block_num -= 1
                    else:
                        small_block_not_selected_list.kernel.append(small_block)
                small_block_list = small_block_not_selected_list
            for qubit_occupation in qubit_occupation_list:
                layer_qubit_occupation = qubit_occupation_combine(qubit_occupation, layer_qubit_occupation)  # list
            current_template = qubit_occupation_template(layer_qubit_occupation, qubit_num, latency)  # list
            if len(qubit_occupation_list) == 0:
                break
        pauli_layers.kernels.append(current_layer)

    # print(pauli_layers)

    return pauli_layers
