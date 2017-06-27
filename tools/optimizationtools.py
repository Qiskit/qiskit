"""
Quantum Optimization tools.

These are simple tools that are used in our optimization examples

Author: Jay Gambetta and Antonio Mezzacapo
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
from tools.pauli import Pauli, label_to_pauli

def SPSA_optimization(obj_fun, initial_theta, SPSA_parameters, max_trials):
    """Minimize the obj_fun(controls).

    initial_theta = the intial controls
    SPSA_parameters = the numerical parameters
    max_trials = the maximum number of trials
    """
    theta_plus = []
    theta_minus = []
    theta = []
    cost_plus = []
    cost_minus = []
    theta.append(initial_theta)
    for k in range(max_trials):
        # SPSA Paramaters
        a_spsa = float(SPSA_parameters[0])/np.power(k+1+SPSA_parameters[4], SPSA_parameters[2])
        c_spsa = float(SPSA_parameters[1])/np.power(k+1, SPSA_parameters[3])
        Delta = 2*np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
        # plus and minus directions
        theta_plus.append(theta[k]+c_spsa*Delta)
        theta_minus.append(theta[k]-c_spsa*Delta)
        # cost fuction for two directions
        cost_plus.append(obj_fun(theta_plus[k])[0])
        cost_minus.append(obj_fun(theta_minus[k])[0])
        if k % 10 == 0:
            print('Energy at theta+ for step # ' + str(k))
            print(cost_plus[k])
            print(('Energy at theta_minus for step # '+str(k)))
            print(cost_minus[k])
        g_spsa = (cost_plus[k] - cost_minus[k])*Delta/(2.0*c_spsa)
        theta.append(theta[k]-a_spsa*g_spsa)

    cost_final = obj_fun(theta[max_trials-1])[0]

    print('Final Energy is\n'+str(cost_final))

    return cost_final, cost_plus, cost_minus, theta_plus, theta_minus


# COST functions
def Energy_Estimate(data, pauli_list):
    """Compute the Energy of a Hamiltonian where P_i elementwise commutes,
    which means that P_i is equilvalent to Z_i.

    H = sum_i alpha_i Z_i

    E = <psi| H |psi>
      = sum_i alpha_i <psi|Z_i|psi>
      = sum_i alphi_i (P_i(up) - P_i(down))


    data = is a dictionary of the form data = {'00000': 10}
    n = number of qubits
    alpha = a vector with elements q0 -- qn

    beta= a matrix of couplings
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
def Energy_Estimate(data, n, alpha, beta):
    """Compute the Energy Estimate of a diagonal Hamiltonian.

    H = alpha_i Z_i + beta_ij Z_i Z_j

    data = is a dictionary of the form data = {'00000': 10}
    n = number of qubits
    alpha = a vector with elements q0 -- qn

    beta= a matrix of couplings
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

def trial_circuit_ry(n, m, theta, entangler_map, meas_string = None, measurement = True):
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
    pauli_string = length of number of qubits string
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    trial_circuit.h(q)
    if meas_string is None:
        meas_string = [None for x in range(n)]
    for i in range(m):
        trial_circuit.barrier(q)
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
            trial_circuit.ry(theta[n * i + j], q[j])
    trial_circuit.barrier(q)
    for j in range(n):
        if meas_string[j] == 'X':
            trial_circuit.h(q[j])
        elif meas_string[j] == 'Y':
            trial_circuit.s(q[j]).inverse()
            trial_circuit.h(q[j])
    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])
    return trial_circuit


def trial_circuit_ryrz(n, m, theta, entangler_map, meas_string = None, measurement = True):
    """Trial function for classical optimization problems.

    n = number of qubits
    m = depth
    theta = control vector of size n*m*2 stacked as theta[n*i*2+2*j+p] where j
    counts the qubits and i the depth and p if y and z.
    entangler_map = {0: [2, 1],
                     1: [2],
                     3: [2],
                     4: [2]}
    control is the key and values are the target
    pauli_string = length of number of qubits string
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    trial_circuit.h(q)
    if meas_string is None:
        meas_string = [None for x in range(n)]
    for i in range(m):
        trial_circuit.barrier(q)
        for node in entangler_map:
            for j in entangler_map[node]:
                trial_circuit.cz(q[node], q[j])
        for j in range(n):
            trial_circuit.ry(theta[n * i * 2 + 2*j], q[j])
            trial_circuit.rz(theta[n * i * 2 + 2*j + 1], q[j])
    trial_circuit.barrier(q)
    for j in range(n):
        if meas_string[j] == 'X':
            trial_circuit.h(q[j])
        elif meas_string[j] == 'Y':
            trial_circuit.s(q[j]).inverse()
            trial_circuit.h(q[j])
    if measurement:
        for j in range(n):
            trial_circuit.measure(q[j], c[j])
    return trial_circuit


def make_Hamiltonian(pauli_list):
        """Compute the Hamiltonian.

        pauli_list is a list of tuples [(number, Pauli(v,w))]
        WARNING. This is exponential in the number of qubits.
        """
        Hamiltonian = 0
        for p in pauli_list:
            Hamiltonian += p[0]*p[1].to_matrix()
        return Hamiltonian


def Hamiltonian_from_file(file_name):
    """Compute the pauli_list from a file name.

    This is a  function at the moment until we can make Hamiltonian
    """
    file = open(file_name, 'r+')
    text_all = file.readlines()
    text_all = [x.strip() for x in text_all]
    pauli_list = []
    for i in range(len(text_all)//2):
        string = text_all[2*i]
        pauli = label_to_pauli(string)
        Numb = float(text_all[2*i+1])
        pauli_list.append((Numb, pauli))
    return pauli_list
