#!/usr/bin/env python

#This is to generate QFT + H + CNOT circuits
#based on circuits explained at Section 4.5 of https://arxiv.org/pdf/1604.06460.pdf
#QFT(n qubits) + H q_0 + CNOT(0, i) for i in 1 ... n

import sys, os
try:
    import qiskit
except ImportError as ierr:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    import qiskit
from qiskit import QuantumProgram
from qiskit.simulators._qasmsimulator import QasmSimulator
import qiskit.qasm as qasm
import qiskit.unroll as unroll

from test_random_qasm_gen import addPrefix

#this is qft as in https://github.com/QISKit/qiskit-sdk-py/blob/master/qiskit/tools/qi/qi.py
from qiskit.tools.qi.qi import qft

if sys.version_info < (3,5):
    raise Exception("Please use Python version 3.5 or greater.")

import math

if __name__ == "__main__":
    #number of qubits
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "nqbits")
        sys.exit(1)

    N = int(sys.argv[1])
    Q_program = QuantumProgram()
    qN = Q_program.create_quantum_register("q", N)
    cN = Q_program.create_classical_register("c", N)
    sName = "qft" + str(N) + "_H_CNOT"
    qftN = Q_program.create_circuit(sName, [qN], [cN])
    #Perform QFT on all qubits
    qft(qftN, qN, N)

    #Perform Hadamard on the first qubits
    qftN.h(qN[0])

    #Perform CNOT on qubit 1 ... N-1 with qubit 0 as control
    for i in range(1, N):
        qftN.cx(qN[0], qN[i])

    #Perform measurements
    for i in range(N):
        qftN.measure(qN[i], cN[i])

    qasm = qftN.qasm()
    outfile = open(sName+".qasm", "w")
    outfile.write(qasm)
    outfile.close()
    print("Writing to "+sName+".qasm")
