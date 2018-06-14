"""
To generate a Bernstein-Vazirani algorithm using 5 qubits, type the following.

python bv_gen.py -q 5 -o bv5
The resulting circuit is stored at bv5.qasm and its drawing at bv5.tex.

For more details, run the above command with -h or --help argument.

@author Raymond Harry Rudy rudyhar@jp.ibm.com
"""
import sys
import numpy as np
import argparse
import random
from qiskit import QuantumProgram
from qiskit.tools.visualization import latex_drawer

if sys.version_info < (3, 5):
    raise Exception("Please use Python 3.5 or later")


def print_qasm(aCircuit, comments=[], outname=None):
    """
        print qasm string with comments
    """
    if outname is None:
        for each in comments:
            print("//"+each)
        print(aCircuit)
    else:
        if not outname.endswith(".qasm"):
            outfilename = outname + ".qasm"
        outfile = open(outfilename, "w")
        for each in comments:
            outfile.write("//"+each)
            outfile.write("\n")
        outfile.write(aCircuit)
        outfile.close()


def draw_circuit(aCircuit, outfilename="bv.tex"):
    """
        draw the circuit
    """
    latex_drawer(aCircuit, outfilename, basis="h,x,cx")


def generate_astring(nqubits, prob=1.0):
    """
        generate a random binary string as a hidden bit string
    """
    answer = []
    for i in range(nqubits):
        if random.random() <= prob:
            answer.append("1")
        else:
            answer.append("0")

    return "".join(answer)


def bin2int(alist):
    """
        convert a binary string into integer
    """
    answer = 0
    temp = alist
    temp.reverse()
    for i in range(len(temp)):
        answer += 2**int(temp[i])
    temp.reverse()
    return answer


def check_astring(astring, nqubits):
    """
        check the validity of string
    """
    if len(astring) > nqubits:
        raise Exception("The length of the hidden string is \
                         longer than the number of qubits")
    else:
        for i in range(len(astring)):
            if astring[i] != "0" and astring[i] != "1":
                raise Exception("Found nonbinary string at "+astring)
    return True


def gen_bv_main(nQubits, hiddenString):
    """
        generate a circuit of the Bernstein-Vazirani algorithm
    """
    Q_program = QuantumProgram()
    # Creating registers
    # qubits for querying the oracle and finding the hidden integer
    qr = Q_program.create_quantum_register("qr", nQubits)
    # for recording the measurement on qr
    cr = Q_program.create_classical_register("cr", nQubits-1)

    circuitName = "BernsteinVazirani"
    bvCircuit = Q_program.create_circuit(circuitName, [qr], [cr])

    # Apply Hadamard gates to the first
    # (nQubits - 1) before querying the oracle
    for i in range(nQubits-1):
        bvCircuit.h(qr[i])

    # Apply 1 and Hadamard gate to the last qubit
    # for storing the oracle's answer
    bvCircuit.x(qr[nQubits-1])
    bvCircuit.h(qr[nQubits-1])

    # Apply barrier so that it is not optimized by the compiler
    bvCircuit.barrier()

    # Apply the inner-product oracle
    hiddenString = hiddenString[::-1]
    for i in range(len(hiddenString)):
        if hiddenString[i] == "1":
            bvCircuit.cx(qr[i], qr[nQubits-1])
    hiddenString = hiddenString[::-1]
    # Apply barrier
    bvCircuit.barrier()

    # Apply Hadamard gates after querying the oracle
    for i in range(nQubits-1):
        bvCircuit.h(qr[i])

    # Measurement
    for i in range(nQubits-1):
        bvCircuit.measure(qr[i], cr[i])

    return Q_program, [circuitName, ]


def main(nQubits, hiddenString, prob, draw, outname):
    if hiddenString is None:
        hiddenString = generate_astring(nQubits-1, prob)
    assert check_astring(hiddenString, nQubits-1) is True, "Invalid hidden str"

    comments = ["Bernstein-Vazirani with " + str(nQubits) + " qubits.",
                "Hidden string is " + hiddenString]
    qp, names = gen_bv_main(nQubits, hiddenString)

    if outname is None:
        outname = "bv_n" + str(nQubits)

    for each in names:
        print_qasm(qp.get_qasm(each), comments, outname)
        if draw:
            if outname is None:
                midfix = "_"+str(nQubits)+"_"+hiddenString
                draw_circuit(qp.get_circuit(each),
                             outfilename=each+midfix+".tex")
            else:
                draw_circuit(qp.get_circuit(each), outfilename=outname+".tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate qasm of \
                                                  the Bernstein-Vazirani \
                                                  algorithm.")
    parser.add_argument("-q", "--qubits", type=int, default=16,
                        help="number of qubits")
    parser.add_argument("-p", "--prob", type=float, default=1.0,
                        help="probability of 1 of the hidden bit")
    parser.add_argument("-a", "--astring", default=None,
                        help="the hidden bitstring")
    parser.add_argument("-s", "--seed", default=0,
                        help="the seed for random number generation")
    parser.add_argument("-d", "--draw", default=False, type=bool,
                        help="flag to draw the circuit")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="output filename")
    args = parser.parse_args()
    # initialize seed
    random.seed(args.seed)
    main(args.qubits, args.astring, args.prob, args.draw, args.output)
