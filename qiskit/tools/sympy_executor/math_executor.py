from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.gate import H, X, Y, Z, S, T, CNOT, IdentityGate, OneQubitGate, UGate
from sympy.core.compatibility import is_sequence, u, unicode, range
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy import pprint, pretty, Matrix, Integer, I, pi, E, Pow, exp, log, Add, sqrt, Mul
from sympy.physics.quantum.qubit import IntQubit

from sympy.physics.quantum.qubit import measure_partial, qubit_to_matrix
from sympy import conjugate, N, re, im
from sympy.physics.quantum import TensorProduct

from extensible_gate_domain import get_sym_op
from circuit_layer_analyzer import gate_qubit_tuples_of_circuit_as_layers

def map_qubit_to_qid(global_qubit_inits):
    qubit2qid = {}
    qid2qubit = {} # reversed mapping
    global_counter = 0
    # important: make sure you assigned the increasing id to q0, q1, q2...
    sorted_global_qubit_inits = sorted(global_qubit_inits) # q[0,1,2,3..] -> 0,1,2,3...
    for qubit in sorted_global_qubit_inits: # assign unique id to each qubit which we will use to refer to the qubit
        if qubit not in qubit2qid:
            qubit2qid[qubit] = global_counter
            qid2qubit[global_counter] = qubit
            global_counter = global_counter + 1
    return qubit2qid, qid2qubit

def pretty_matrix_qubit(q):
    return pretty(represent(q)) # represent: q -> matrix

# note: this applies to a single sympy operation, but not to a multiplication of operations.
# we do not really need to know the matrix details.
def pretty_matrix_sympy_op(sym_op): # not exactly as gate, gate is like class and op is like object
    return pretty(sym_op.get_target_matrix(format='sympy'))


def ket(input):
    return '|' + input + '>'

def label(i, nqubits): # from lower-level wire (q[3]) to higer level wire q[0])
    qubit_array = [int(i & (1 << x) != 0) for x in range(nqubits)]
    qubit_array.reverse()
    return ket("".join([str(tmp) for tmp in qubit_array]))

def getHeaderKet(qid2qubit, partial_list=None):
    header = ",".join([qid2qubit[k] for k in sorted(qid2qubit, reverse=True) if partial_list == None or k in partial_list])
    header = ket(header)
    return header

def matrix_with_header(qm, headerlabel):
    N = qm.shape[0]
    nqubits = log(N, 2)
    summary = ""
    maxlength = max([len(str(qm[i,0])) for i in range(N)])
    formatstr = '|{:'+ str(maxlength)+'s}|'
    headerformatstr = ' {:'+ str(maxlength)+'s} '

    for i in range(N):
        tmp = formatstr.format(str(qm[i,0])) + " "+ label(i, nqubits) # label in order q[3,2,1,0...]
        summary = summary + "\n" + tmp

    # wires_loop = sorted(wires_loop, key=lambda tup:(tup[0], -tup[1]), reverse=True) # important: prevent "q[2] is above q[1]"
    header = headerformatstr.format("") + " "+ headerlabel
    return header + summary


# def layer_state_readable_summary(q, qid2qubit):
#     # if not withlabel:
#     #     summary = pretty_matrix_qubit(q)
#     #     return getHeaderKet(qid2qubit) + "\n" + summary
#     # else:
#
#     qm = represent(q) # Nx1 2-d matrix for representing a N-entry vector
#     headerlabel = getHeaderKet(qid2qubit)
#     return matrix_with_header(qm, headerlabel)


#<type 'tuple'>: (-3/8, I, sqrt(2), |1101110>)
def collapseIfNecessary(myargs):
    if len(myargs) == 2:
        return myargs
    elif len(myargs) == 3:
        tmp = []
        tmp.append(myargs[0] * myargs[1])
        tmp.append(myargs[2])
        return tuple(tmp)
    elif len(myargs) == 4:
        tmp = []
        tmp.append(myargs[0] * myargs[1]*myargs[2])
        tmp.append(myargs[3])
        return tuple(tmp)
    elif len(myargs) == 5:
        tmp = []
        tmp.append(myargs[0] * myargs[1]*myargs[2]*myargs[3])
        tmp.append(myargs[4])
        return tuple(tmp)
    else:
        print("tuple that has problem:" + " " + myargs)
        return NotImplemented

def make_represent_of_imaginary_qubit(superposition, to_measure_list):
    def index2qid(index, nqubits):
        return nqubits-1-index

    def multiply_conjugate(x): # better than x**2 considering we have complex coefficient
        return im(x)**2 + re(x)**2

    #print superposition
    #-3*|00000>/16 - 3*|00001>/16 - 3*|00010>/16 - 3*|00011>/16 - 3*|00100>/16 - 3*|00101>/16 - 3*|00110>/16 - 3*|00111>/16 - 3*|01000>/16 - 3*|01001>/16 - 3*|01010>/16 - 3*|01011>/16 - 3*|01100>/16 - 3*|01101>/16 - 3*|01110>/16 - 11*|01111>/16
    #print to_measure_list
    #[0, 1, 2, 3]


    coefficient_qubit_list = []
    qubitprojected2prob = {}
    if isinstance(superposition, Add):
        for addpart in superposition._args:
            if isinstance(addpart, Mul):
                coefficient_qubit_list.append(collapseIfNecessary(addpart._args))         # (1/8, sqrt(2), |11110>)
            elif isinstance(addpart, Qubit):
                coefficient_qubit_list.append((1,addpart))
            else:
                raise Exception("beyond my knowledge")
    elif isinstance(superposition, Mul):
        coefficient_qubit_list.append(collapseIfNecessary(superposition._args))
    elif isinstance(superposition, Qubit):
        coefficient_qubit_list.append((1,superposition))
    else:
        raise Exception("beyond my knowledge")


    for coefficient_qubit in coefficient_qubit_list:
        amplitude = coefficient_qubit[0]
        qubit = coefficient_qubit[1]
        nqubits = qubit.nqubits
        qubit_projected = [qubit.qubit_values[i] for i in range(nqubits) if index2qid(i, nqubits) in to_measure_list]
        qubit_projected = Qubit(*tuple(qubit_projected))
        if qubit_projected not in qubitprojected2prob:
            qubitprojected2prob[qubit_projected] = multiply_conjugate(amplitude)
        else:
            qubitprojected2prob[qubit_projected] += multiply_conjugate(amplitude)

    imagineQ = None
    for qubit in qubitprojected2prob:
        prob = qubitprojected2prob[qubit]
        amplitude = sqrt(prob)
        if imagineQ == None:
            imagineQ = amplitude*qubit
        else:
            imagineQ += amplitude*qubit


    imagineQMatrix = qubit_to_matrix(imagineQ, format='sympy') # convert add/mul superposition to matrix
    return imagineQMatrix




# return represent(q), label
# check-equivalence will first check label, then check the matrix represent(q)
def math_execute(args, circuit, states_per_step=None):
    schedule_list_of_layers, global_qubit_inits, _ = gate_qubit_tuples_of_circuit_as_layers(circuit)
    # schedule_list_of_layers:
    #[[('H', 'q1'), ('H', 'q2')], [('H', 'q2')], [('CX', 'q1', 'q2')], [('H', 'q2')], [('H', 'q1'), ('H', 'q2')], [('X', 'q1'), ('X', 'q2')], [('H', 'q2')], [('CX', 'q1', 'q2')], [('H', 'q2')], [('X', 'q1'), ('X', 'q2')], [('H', 'q1'), ('H', 'q2')]]

    qubit2qid, qid2qubit = map_qubit_to_qid(global_qubit_inits)

    q = Qubit(*tuple([0]*len(qubit2qid)))
    to_measure_list = []
    layerID = 0
    for layer in schedule_list_of_layers:
        layerID = layerID + 1
        _layer_op = None
        _layer_op_fake_for_name = None
        for gate_qubit_tuple in layer:
            _gate = gate_qubit_tuple[0] # to avoid name collision, I will prefix my variable with _
            _qubits = gate_qubit_tuple[1:] # may be one or more
            _qids = [qubit2qid[_qubit] for _qubit in _qubits]

            if _gate.upper() == "MEASURE":
                # this wire becomes dead, meaning you cannot add more gates from now to the end.
                # Hence, let us delay the measure to the end
                to_measure_list.extend(_qids)
                continue

            _sym_op = get_sym_op(_gate.upper(), tuple(_qids))
            _sym_op_fake_for_name = get_sym_op(_gate.upper(), _qubits)
            if _layer_op == None:
                _layer_op = _sym_op
                _layer_op_fake_for_name = _sym_op_fake_for_name
            else:
                _layer_op = _layer_op * _sym_op # in FIFO order
                _layer_op_fake_for_name = _layer_op_fake_for_name * _sym_op_fake_for_name

        # after each depth/step:
        if _layer_op != None:
            if 'optimization_level' not in vars(args) and args.report_after_every_step:
                print("\n*********state after step:"+ str(layerID) + "*********")
                print("layer summary:\n", _layer_op_fake_for_name)
            _layer_op = _layer_op * q
            q = qapply(_layer_op)

            #peng: recording state after each step
            if states_per_step != None:
                astate = matrix_with_header(represent(q), getHeaderKet(qid2qubit))
                states_per_step[str(layerID)] = astate

            if 'optimization_level' not in vars(args) and args.report_after_every_step:
                print("state summary:\n" + " ")
                print(matrix_with_header(represent(q), getHeaderKet(qid2qubit)))


    # return utility:
    if args.return_measured_state and len(to_measure_list) != 0: # if users forgot to provide measure, we assume all qubits should be considered by default
        matrix = make_represent_of_imaginary_qubit(q, to_measure_list)
        header = getHeaderKet(qid2qubit, to_measure_list)
        if 'optimization_level' not in vars(args):
            print("\n*********final measured state:\n" + " " +  matrix_with_header(matrix, header))
        return matrix, header, matrix_with_header(matrix, header)

    else:
        matrix = represent(q)
        header = getHeaderKet(qid2qubit)

        if 'optimization_level' not in vars(args):
            print("\n*********final complete state:\n" + " " + matrix_with_header(matrix, header))

        return matrix, header, matrix_with_header(matrix, header)
        # matrix+label is better than q itself which has same information as matrix
































from sympy.physics.mechanics import ReferenceFrame, Vector, dot
from sympy import conjugate, N, re, im

if __name__ == "__main__":

    y = E**(I*pi/4)
    print(y)
    x =  1/4*I*sqrt(2)*y
    print(im(x)**2 + re(x)**2)


    print(label(6, 3))

    # q = Qubit("q0", [0,1])
    # print q.apply(Gates.H)
    print(E**(I*pi/4))

    q = Qubit(0, 0)

    print(qapply(H(0) * H(1) * q))
    # psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    # print represent(psi) # matrix rep of qubit
    #
    #
    #
    HMatrix = represent(H(0).get_target_matrix(format='sympy')) # matrix rep of gate
    pprint(HMatrix)
    pprint(H(0).get_target_matrix(format='sympy'))
    p = TensorProduct(HMatrix, HMatrix)
    print(p)


