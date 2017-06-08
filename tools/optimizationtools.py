"""
Quantum Optimization tools.

These are simple tools that are used in our optimization examples

Author: Jay Gambetta
"""
import sys
import os
# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, ry, barrier, cz
from qiskit.simulators._simulatortools import enlarge_single_opt, enlarge_two_opt
import numpy as np


def cost_function(data, n, alpha, beta):
    """Compute the cost function.

    data  is a dictionary of the form data = {'00000': 10}
    n = number of qubits
    alpha is a vector with elements q0 -- qn
    beta is a matrix of couplings
    """

    temp = 0
    tot = sum(data.values())
    for key in data:
        observable = 0
        for j in range(len(key) - n, len(key)):
            if key[j] == '0':
                observable = observable + alpha[len(key) - 1 - j]
            elif key[j] == '1':
                observable = observable - alpha[len(key) - 1 - j]
            for i in range(j):
                if key[j] == '0' and key[i] == '0':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '1':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '0' and key[i] == '1':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '0':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
            for i in range(j + 1, len(key)):
                if key[j] == '0' and key[i] == '0':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '1':
                    observable = observable + \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '0' and key[i] == '1':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
                elif key[j] == '1' and key[i] == '0':
                    observable = observable - \
                        beta[len(key) - 1 - i, len(key) - 1 - j]
        temp += data[key] * observable / tot
    return temp


def make_Hamiltonian(n, alpha, beta):
    """Compute the cost function.

    n = number of qubits
    alpha is a vector with elements q0 -- qn
    beta is a matrix of couplings
    WARNING. This is exponential in the number of qubits.
    """

    Hamiltonian = 0
    Z = np.array([[1, 0], [0, -1]])
    ZZ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    for j in range(n):
        Hamiltonian += alpha[j]*enlarge_single_opt(Z, j, n)
        for i in range(0, j):
            Hamiltonian += beta[i, j]*enlarge_two_opt(ZZ, i, j, n)

    return Hamiltonian


def trial_funtion_optimization(n, m, theta, entangler_map):
    """Trial function for classical optimization problems.

    n = number of qubits
    m = depth
    theta = control vector of size n*m stacked as theta[n*i+j] where j counts
           the qubits and i the depth
    entangler_map = {0: [2, 1],
                     1: [2],
                     3: [2],
                     4: [2]}
    control is the key and values are the target
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    trial_circuit.h(q)
    for i in range(m):
        trial_circuit.barrier(q)
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
            trial_circuit.ry(theta[n * i + j], q[j])
    trial_circuit.barrier(q)
    for j in range(n):
        trial_circuit.measure(q[j], c[j])
    return trial_circuit


def trial_funtion_optimization_no_meas(n, m, theta, entangler_map):
    """Trial function for classical optimization problems.

    n = number of qubits
    m = depth
    theta = control vector of size n*m stacked as theta[n*i+j] where j counts
           the qubits and i the depth
    entangler_map = {0: [2, 1],
                     1: [2],
                     3: [2],
                     4: [2]}
    control is the key and values are the target
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    trial_circuit.h(q)
    for i in range(m):
        trial_circuit.barrier(q)
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
            trial_circuit.ry(theta[n * i + j], q[j])
    return trial_circuit
