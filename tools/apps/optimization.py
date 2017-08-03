# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Quantum Optimization tools.

These are simple tools that are used in our optimization examples
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, ry, barrier, cz
import numpy as np
from tools.qi.pauli import Pauli, label_to_pauli


def SPSA_optimization(obj_fun, initial_theta, SPSA_parameters, max_trials, save_steps = 1,last_avg=1):
    """Minimize the obj_fun(controls).

    initial_theta = the intial controls
    SPSA_parameters = the numerical parameters
    max_trials = the maximum number of trials
    """
    theta_plus_save = []
    theta_minus_save = []
    cost_plus_save = []
    cost_minus_save = []
    theta = initial_theta
    theta_best=np.zeros(initial_theta.shape)
    for k in range(max_trials):
        # SPSA Paramaters
        a_spsa = float(SPSA_parameters[0])/np.power(k+1+SPSA_parameters[4], SPSA_parameters[2])
        c_spsa = float(SPSA_parameters[1])/np.power(k+1, SPSA_parameters[3])
        Delta = 2*np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
        # plus and minus directions
        theta_plus = theta + c_spsa*Delta
        theta_minus = theta - c_spsa*Delta
        # cost fuction for two directions
        cost_plus = obj_fun(theta_plus)[0]
        cost_minus = obj_fun(theta_minus)[0]
        # derivative estimate
        g_spsa = (cost_plus - cost_minus)*Delta/(2.0*c_spsa)
        # updated theta
        theta = theta - a_spsa*g_spsa
        # saving
        if k % save_steps == 0:
            print('objective function at theta+ for step # ' + str(k))
            print(cost_plus)
            print(('objective function at theta- for step # '+str(k)))
            print(cost_minus)
            theta_plus_save.append(theta_plus)
            theta_minus_save.append(theta_minus)
            cost_plus_save.append(cost_plus)
            cost_minus_save.append(cost_minus)

        if k>=max_trials-last_avg:
            theta_best+=theta/last_avg

    # final cost update
    cost_final = obj_fun(theta_best)[0]
    print('Final objective function is: ' + str(cost_final))
    return cost_final, theta_best, cost_plus_save, cost_minus_save, theta_plus_save, theta_minus_save


def SPSA_calibration(obj_fun, initial_theta, initial_c, target_update, stat):
    """Calibrates the first SPSA parameter.
    The calibration is chosen such that the first theta update is on average
    (with statistics regulated by stat) equivalent to target_update, given
    an initial_c (=SPSA_parameters[1]) value.

    Returns all 5 SPSA_parameters:

    SPSA_parameters[0] -> calibrated
    SPSA_parameters[1] -> input by user (initial_c)
    SPSA_parameters[2] -> fixed at 0.602
    SPSA_parameters[3] -> fixed at 0.101
    SPSA_parameters[4] -> fixed at 0
    """

    SPSA_parameters = np.zeros((5))
    SPSA_parameters[1] = initial_c
    SPSA_parameters[2] = 0.602
    SPSA_parameters[3] = 0.101
    SPSA_parameters[4] = 0

    Delta_obj = 0
    for i in range(stat):

        if i % 5 == 0:
            print('calibration step # '+str(i)+' of '+str(stat))

        Delta = 2*np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
        obj_plus = obj_fun(initial_theta+initial_c*Delta)[0]
        obj_minus = obj_fun(initial_theta-initial_c*Delta)[0]
        Delta_obj += np.absolute(obj_plus - obj_minus)/stat

    SPSA_parameters[0] = target_update*2/Delta_obj*SPSA_parameters[1]*(SPSA_parameters[4]+1)

    print('calibrated SPSA_parameters[0] is '+str(SPSA_parameters[0]))

    return SPSA_parameters


# COST functions
def Measure_pauli_z(data, pauli):
    """Compute the expectation value of Z.

    Z is represented by Z^v where v has lenght number of qubits and is 1
    if Z is present and 0 otherwise.

    data = is a dictionary of the form data = {'00000': 10}

    M = <psi|Z^v|psi>
      = \sum_lambda  lambda  |<lambda |psi>|^2
      = sum_i lambda(i) P(i)
      where i is the bitstring (key of data)
      = sum_key lambda(key) #key/total_values
    """
    observable = 0
    tot = sum(data.values())
    for key in data:
        value = 1
        for j in range(pauli.numberofqubits):

            if (pauli.v[j] == 1 or pauli.w[j] == 1) and key[pauli.numberofqubits - j - 1] == '1':
                    value = -value

        observable = observable + value*data[key]/tot
    return observable


def Energy_Estimate(data, pauli_list):
        """Compute expectation value of a list of Paulis with coefficients.

        pauli_list is a list of tuples [(number, Pauli(v,w))]
        """
        energy = 0
        if np.ndim(pauli_list) == 1:
            energy = pauli_list[0]*Measure_pauli_z(data, pauli_list[1])
        else:
            for p in pauli_list:
                energy += p[0]*Measure_pauli_z(data, p[1])
        return energy


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
    meas_string = the pauli to be measured
    measurement = true/false if measurement is to be done
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


def trial_circuit_computational(n, state, meas_string = None, measurement = True):
    """Trial function for classical optimization problems.

    n = number of qubits
    state = a bit string for the state prepared.
    meas_string = the pauli to be measured
    measurement = true/false if measurement is to be done
    """
    q = QuantumRegister("q", n)
    c = ClassicalRegister("c", n)
    trial_circuit = QuantumCircuit(q, c)
    if meas_string is None:
        meas_string = [None for x in range(n)]
    if len(state) == n:
        for j in range(n):
            if state[n-j-1] == "1":
                trial_circuit.x(q[j])
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

        pauli_list is a list of tuples [(coefficient, Pauli(v,w))]
        WARNING. This is exponential in the number of qubits.
        """
        Hamiltonian = 0
        for p in pauli_list:
            Hamiltonian += p[0]*p[1].to_matrix()
        return Hamiltonian


def Hamiltonian_from_file(file_name):
    """Compute the pauli_list from a file."""
    file = open(file_name, 'r+')
    ham_array = file.readlines()
    ham_array = [x.strip() for x in ham_array]
    pauli_list = []
    for i in range(len(ham_array)//2):
        pauli = label_to_pauli(ham_array[2*i])
        Numb = float(ham_array[2*i+1])
        pauli_list.append([Numb, pauli])
    return pauli_list
