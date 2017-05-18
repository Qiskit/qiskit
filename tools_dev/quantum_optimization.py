"""
Quantum Optimization tools.

These are simple methods for common tasks in our optimization.

Author: Jay Gambetta
"""
import sys
sys.path.append("..")
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, ry, barrier, cz


def cost_classical(data, n, alpha, beta):
    """Compute the cost function.

    n = number of qubits
    alpha is a vector with elements q0 -- qn
    beta is a matrix of couplings

    NOTE THIS SHOULD BE MADE TO WORK WITH THE UPPER TRIANGLE.
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
