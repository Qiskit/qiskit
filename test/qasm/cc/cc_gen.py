"""
To generate a circuit for counterfeit-coin finding
algorithm using 15 coins and the false coin is the third coin,
type the following.

python cc_gen.py -c 15 -f 3

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


def generate_false(nCoins):
    """
        generate a random index of false coin (counting from zero)
    """
    return random.randint(0, nCoins-1)


def gen_cc_main(nCoins, indexOfFalseCoin):
    """
        generate a circuit of the counterfeit coin problem
    """
    Q_program = QuantumProgram()
    # using the last qubit for storing the oracle's answer
    nQubits = nCoins + 1
    # Creating registers
    # qubits for querying coins and storing the balance result
    qr = Q_program.create_quantum_register("qr", nQubits)
    # for recording the measurement on qr
    cr = Q_program.create_classical_register("cr", nQubits)

    circuitName = "CounterfeitCoinProblem"
    ccCircuit = Q_program.create_circuit(circuitName, [qr], [cr])

    # Apply Hadamard gates to the first nCoins quantum register
    # create uniform superposition
    for i in range(nCoins):
        ccCircuit.h(qr[i])

    # check if there are even number of coins placed on the pan
    for i in range(nCoins):
        ccCircuit.cx(qr[i], qr[nCoins])

    # perform intermediate measurement to check if the last qubit is zero
    ccCircuit.measure(qr[nCoins], cr[nCoins])

    # proceed to query the quantum beam balance if cr is zero
    ccCircuit.x(qr[nCoins]).c_if(cr, 0)
    ccCircuit.h(qr[nCoins]).c_if(cr, 0)

    # we rewind the computation when cr[N] is not zero
    for i in range(nCoins):
        ccCircuit.h(qr[i]).c_if(cr, 2**nCoins)

    # apply barrier for marking the beginning of the oracle
    ccCircuit.barrier()

    ccCircuit.cx(qr[indexOfFalseCoin], qr[nCoins]).c_if(cr, 0)

    # apply barrier for marking the end of the oracle
    ccCircuit.barrier()

    # apply Hadamard gates to the first nCoins qubits
    for i in range(nCoins):
        ccCircuit.h(qr[i]).c_if(cr, 0)

    # measure qr and store the result to cr
    for i in range(nCoins):
        ccCircuit.measure(qr[i], cr[i])

    return Q_program, [circuitName, ]


def main(nCoins, falseIndex, draw, outname):
    comments = ["Counterfeit coin finding with " + str(nCoins) + " coins.",
                "The false coin is " + str(falseIndex)]
    if outname is None:
        outname = "cc_n" + str(nCoins + 1)
    qp, names = gen_cc_main(nCoins, falseIndex)
    for each in names:
        print_qasm(qp.get_qasm(each), comments, outname)
        if draw:
            if outname is None:
                midfix = "_"+str(nCoins)+"_"+str(falseIndex)
                draw_circuit(qp.get_circuit(each),
                             outfilename=each+midfix+".tex")
            else:
                draw_circuit(qp.get_circuit(each),
                             outfilename=outname+".tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate qasm of \
                                                  the counterfeit-coin \
                                                  finding algorithm.")
    parser.add_argument("-c", "--coins", type=int, default=16,
                        help="number of coins")
    parser.add_argument("-f", "--false", type=int, default=None,
                        help="index of false coin")
    parser.add_argument("-s", "--seed", default=0,
                        help="the seed for random number generation")
    parser.add_argument("-d", "--draw", default=False, type=bool,
                        help="flag to draw the circuit")
    parser.add_argument("-o", "--output", default=None, type=str,
                        help="output filename")
    args = parser.parse_args()
    # initialize seed
    random.seed(args.seed)

    if args.false is None:
        args.false = generate_false(args.coins)
    main(args.coins, args.false, args.draw, args.output)
