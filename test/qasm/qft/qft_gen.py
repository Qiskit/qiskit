"""
Generate circuits of Quantum Fourier Transform for Quantum Volume analysis.

Example run:
  python qft.py -n 5
"""

import sys
import math
import time
import argparse
import numpy as np
from scipy.linalg import qr
from scipy.linalg import det
from qiskit import QuantumProgram
from qiskit.mapper import two_qubit_kak

if sys.version_info < (3, 0):
    raise Exception("Please use Python version 3 or greater.")


def cu1(circ, l, a, b):
    circ.u1(l/2, a)
    circ.cx(a, b)
    circ.u1(-l/2, b)
    circ.cx(a, b)
    circ.u1(l/2, b)


def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            cu1(circ, math.pi/float(2**(j-k)), q[j], q[k])
#            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])


def build_model_circuits(name, n):
    qp = QuantumProgram()
    q = qp.create_quantum_register("q", n)
    c = qp.create_classical_register("c", n)

    qftcirc = qp.create_circuit("meas", [q], [c])

    qft(qftcirc, q, n)
    qftcirc.barrier(q)
    for j in range(n):
        qftcirc.measure(q[j], c[j])

    qp.add_circuit("%s_%d" % (name, n), qftcirc)

    return qp


def main():
    parser = argparse.ArgumentParser(description="Create circuits \
                                                  of Quantum Fourier \
                                                  Transform for \
                                                  quantum volume analysis.")
    parser.add_argument('--name', default='qft', help='circuit name')
    parser.add_argument('-n', '--qubits', default=5,
                        type=int, help='number of circuit qubits')
    args = parser.parse_args()

    qp = build_model_circuits(name=args.name, n=args.qubits)

    circuit_name = args.name+'_n'+str(args.qubits)
    f = open(circuit_name+'.qasm', 'w')
    f.write(qp.get_qasm(name="%s_%d" % (args.name, args.qubits)))
    f.close()


if __name__ == "__main__":
    main()
