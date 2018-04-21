# -*- coding: utf-8 -*-
# pylint: disable=unused-import,invalid-name

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
These are tools that are used in the classical optimization and chemistry
tutorials
"""
import uuid
import copy
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, ry, barrier, cz, x, y, z
from qiskit.tools.qi.pauli import Pauli, label_to_pauli


def SPSA_optimization(obj_fun, initial_theta, SPSA_parameters, max_trials,
                      save_steps=1, last_avg=1):
    """Minimizes obj_fun(theta) with a simultaneous perturbation stochastic
    approximation algorithm.

    Args:
        obj_fun (callable): the function to minimize
        initial_theta (numpy.array): initial value for the variables of
            obj_fun
        SPSA_parameters (list[float]) :  the parameters of the SPSA
            optimization routine
        max_trials (int) : the maximum number of trial steps ( = function
            calls/2) in the optimization
        save_steps (int) : stores optimization outcomes each 'save_steps'
            trial steps
        last_avg (int) : number of last updates of the variables to average
            on for the final obj_fun
    Returns:
        list: a list with the following elements:
            cost_final : final optimized value for obj_fun
            theta_best : final values of the variables corresponding to
                cost_final
            cost_plus_save : array of stored values for obj_fun along the
                optimization in the + direction
            cost_minus_save : array of stored values for obj_fun along the
                optimization in the - direction
            theta_plus_save : array of stored variables of obj_fun along the
                optimization in the + direction
            theta_minus_save : array of stored variables of obj_fun along the
                optimization in the - direction
    """

    theta_plus_save = []
    theta_minus_save = []
    cost_plus_save = []
    cost_minus_save = []
    theta = initial_theta
    theta_best = np.zeros(initial_theta.shape)
    for k in range(max_trials):
        # SPSA Paramaters
        a_spsa = float(SPSA_parameters[0]) / np.power(k + 1 +
                                                      SPSA_parameters[4],
                                                      SPSA_parameters[2])
        c_spsa = float(SPSA_parameters[1]) / np.power(k + 1,
                                                      SPSA_parameters[3])
        delta = 2 * np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
        # plus and minus directions
        theta_plus = theta + c_spsa * delta
        theta_minus = theta - c_spsa * delta
        # cost fuction for the two directions
        cost_plus = obj_fun(theta_plus)
        cost_minus = obj_fun(theta_minus)
        # derivative estimate
        g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
        # updated theta
        theta = theta - a_spsa * g_spsa
        # saving
        if k % save_steps == 0:
            print('objective function at theta+ for step # ' + str(k))
            print("%.7f" % cost_plus)
            print(('objective function at theta- for step # ' + str(k)))
            print("%.7f" % cost_minus)
            theta_plus_save.append(theta_plus)
            theta_minus_save.append(theta_minus)
            cost_plus_save.append(cost_plus)
            cost_minus_save.append(cost_minus)

        if k >= max_trials - last_avg:
            theta_best += theta / last_avg

    # final cost update
    cost_final = obj_fun(theta_best)
    print('Final objective function is: %.7f' % cost_final)
    return [cost_final, theta_best, cost_plus_save, cost_minus_save,
            theta_plus_save, theta_minus_save]


def SPSA_calibration(obj_fun, initial_theta, initial_c, target_update, stat):
    """Calibrates and returns the SPSA parameters.

    Args:
        obj_fun (callable): the function to minimize.
        initial_theta (numpy.array): initial value for the variables of
            obj_fun.
        initial_c (float) : first perturbation of intitial_theta.
        target_update (float) : the aimed update of variables on the first
            trial step.
        stat (int) : number of random gradient directions to average on in
            the calibration.
    Returns:
        numpy.array: An array of 5 SPSA_parameters to use in the optimization.
    """

    SPSA_parameters = np.zeros((5))
    SPSA_parameters[1] = initial_c
    SPSA_parameters[2] = 0.602
    SPSA_parameters[3] = 0.101
    SPSA_parameters[4] = 0
    delta_obj = 0
    for i in range(stat):
        if i % 5 == 0:
            print('calibration step # ' + str(i) + ' of ' + str(stat))

        delta = 2 * np.random.randint(2, size=np.shape(initial_theta)[0]) - 1
        obj_plus = obj_fun(initial_theta + initial_c * delta)
        obj_minus = obj_fun(initial_theta - initial_c * delta)
        delta_obj += np.absolute(obj_plus - obj_minus) / stat

    SPSA_parameters[0] = target_update * 2 / delta_obj \
        * SPSA_parameters[1] * (SPSA_parameters[4] + 1)

    print('calibrated SPSA_parameters[0] is %.7f' % SPSA_parameters[0])

    return SPSA_parameters


def measure_pauli_z(data, pauli):
    """Compute the expectation value of Z.

    Z is represented by Z^v where v has lenght number of qubits and is 1
    if Z is present and 0 otherwise.

    Args:
        data (dict): a dictionary of the form data = {'00000': 10}
        pauli (Pauli): a Pauli object
    Returns:
        float: Expected value of pauli given data
 """
    observable = 0
    tot = sum(data.values())
    for key in data:
        value = 1
        for j in range(pauli.numberofqubits):
            if ((pauli.v[j] == 1 or pauli.w[j] == 1) and
                    key[pauli.numberofqubits - j - 1] == '1'):
                value = -value

        observable = observable + value * data[key] / tot
    return observable


def Energy_Estimate(data, pauli_list):
    """Compute expectation value of a list of diagonal Paulis with
    coefficients given measurement data. If somePaulis are non-diagonal
    appropriate post-rotations had to be performed in the collection of data

    Args:
        data (dict): output of the execution of a quantum program
        pauli_list (list): list of [coeff, Pauli]
    Returns:
        float: The expectation value
    """
    energy = 0
    if np.ndim(pauli_list) == 1:
        energy = pauli_list[0] * measure_pauli_z(data, pauli_list[1])
    else:
        for p in pauli_list:
            energy += p[0] * measure_pauli_z(data, p[1])
    return energy


def index_2_bit(state_index, num_bits):
    """Returns bit string corresponding to quantum state index

    Args:
        state_index (int): basis index of a quantum state
        num_bits (int): the number of bits in the returned string
    Returns:
        numpy.array: A integer array with the binary representation of
            state_index
    """
    return np.array([int(c) for c
                     in np.binary_repr(state_index, num_bits)[::-1]],
                    dtype=np.uint8)


def group_paulis(pauli_list):
    """
    Groups a list of (coeff,Pauli) tuples into tensor product basis (tpb) sets

    Args:
        pauli_list (list): a list of (coeff, Pauli object) tuples.
    Returns:
        list: A list of tpb sets, each one being a list of (coeff, Pauli
            object) tuples.
    """
    n = len(pauli_list[0][1].v)
    pauli_list_grouped = []
    pauli_list_sorted = []
    for p_1 in pauli_list:
        if p_1 not in pauli_list_sorted:
            pauli_list_temp = []
            # pauli_list_temp.extend(p_1) # this is going to signal the total
            # post-rotations of the set (set master)
            pauli_list_temp.append(list(p_1))
            pauli_list_temp.append(copy.deepcopy(list(p_1)))
            pauli_list_temp[0][0] = 0
            for p_2 in pauli_list:
                if p_2 not in pauli_list_sorted and p_1[1] != p_2[1]:
                    j = 0
                    for i in range(n):
                        if not ((p_2[1].v[i] == 0 and p_2[1].w[i] == 0) or
                                (p_1[1].v[i] == 0 and p_1[1].w[i] == 0) or
                                (p_2[1].v[i] == p_1[1].v[i] and
                                 p_2[1].w[i] == p_1[1].w[i])):
                            break
                        else:
                            # update master
                            if p_2[1].v[i] == 1 or p_2[1].w[i] == 1:
                                pauli_list_temp[0][1].v[i] = p_2[1].v[i]
                                pauli_list_temp[0][1].w[i] = p_2[1].w[i]

                        j += 1

                    if j == n:
                        pauli_list_temp.append(p_2)
                        pauli_list_sorted.append(p_2)
            pauli_list_grouped.append(pauli_list_temp)

    return pauli_list_grouped


def print_pauli_list_grouped(pauli_list_grouped):
    """Print a list of Pauli operators which has been grouped into tensor
    product basis (tpb) sets.

    Args:
        pauli_list_grouped (list of lists of (coeff, pauli) tuples): the
            list of Pauli operators grouped into tpb sets
    """
    for i, _ in enumerate(pauli_list_grouped):
        print('Post Rotations of TPB set ' + str(i) + ':')
        print(pauli_list_grouped[i][0][1].to_label())
        print(str(pauli_list_grouped[i][0][0]) + '\n')
        for j in range((len(pauli_list_grouped[i]) - 1)):
            print(pauli_list_grouped[i][j + 1][1].to_label())
            print("%.7f" % pauli_list_grouped[i][j + 1][0])

        print('\n')


def eval_hamiltonian(Q_program, hamiltonian, input_circuit, shots, device):
    """Calculates the average value of a Hamiltonian on a state created by the
     input circuit

    Args:
        Q_program (QuantumProgram): QuantumProgram object used to run the
            input circuit.
        hamiltonian (array or matrix or list): a representation of the
            Hamiltonian or observables to be measured. If it is a list, it is
            a list of Pauli operators grouped into tpb sets.
        input_circuit (QuantumCircuit): input circuit.
        shots (int): number of shots considered in the averaging. If 1 the
            averaging is exact.
        device (str): the backend used to run the simulation.
    Returns:
        float: Average value of the Hamiltonian or observable.
    """
    energy = 0

    if shots == 1:
        # Hamiltonian is not a pauli_list grouped into tpb sets
        if not isinstance(hamiltonian, list):
            circuit = ['c' + str(uuid.uuid4())]    # unique random circuit for no collision
            Q_program.add_circuit(circuit[0], input_circuit)
            result = Q_program.execute(circuit, device, shots=shots,
                                       config={"data": ["quantum_state"]})
            quantum_state = result.get_data(circuit[0]).get('quantum_state')
            if quantum_state is None:
                quantum_state = result.get_data(
                    circuit[0]).get('quantum_states')
                if quantum_state:
                    quantum_state = quantum_state[0]
            # Diagonal Hamiltonian represented by 1D array
            if (hamiltonian.shape[0] == 1 or
                    np.shape(np.shape(np.array(hamiltonian))) == (1,)):
                energy = np.sum(hamiltonian * np.absolute(quantum_state) ** 2)
            # Hamiltonian represented by square matrix
            elif hamiltonian.shape[0] == hamiltonian.shape[1]:
                energy = np.inner(np.conjugate(quantum_state),
                                  np.dot(hamiltonian, quantum_state))
        # Hamiltonian represented by a Pauli list
        else:
            circuits = []
            circuits_labels = []
            circuits.append(input_circuit)
            # Trial circuit w/o the final rotations
            circuits_labels.append('circuit_label0' + str(uuid.uuid4()))
            Q_program.add_circuit(circuits_labels[0], circuits[0])
            # Execute trial circuit with final rotations for each Pauli in
            # hamiltonian and store from circuits[1] on
            n_qubits = input_circuit.regs['q'].size
            q = QuantumRegister(n_qubits, "q")
            i = 1
            for p in hamiltonian:
                circuits.append(copy.deepcopy(input_circuit))
                for j in range(n_qubits):
                    if p[1].v[j] == 0 and p[1].w[j] == 1:
                        circuits[i].x(q[j])
                    elif p[1].v[j] == 1 and p[1].w[j] == 0:
                        circuits[i].z(q[j])
                    elif p[1].v[j] == 1 and p[1].w[j] == 1:
                        circuits[i].y(q[j])

                circuits_labels.append('circuit_label' + str(i) + str(uuid.uuid4()))
                Q_program.add_circuit(circuits_labels[i], circuits[i])
                i += 1
            result = Q_program.execute(circuits_labels, device, shots=shots)
            # no Pauli final rotations
            quantum_state_0 = result.get_data(
                circuits_labels[0])['quantum_state']
            i = 1
            for p in hamiltonian:
                quantum_state_i = result.get_data(
                    circuits_labels[i])['quantum_state']
                # inner product with final rotations of (i-1)-th Pauli
                energy += p[0] * np.inner(np.conjugate(quantum_state_0),
                                          quantum_state_i)
                i += 1
    # finite number of shots and hamiltonian grouped in tpb sets
    else:
        circuits = []
        circuits_labels = []
        n = int(len(hamiltonian[0][0][1].v))
        q = QuantumRegister(n, "q")
        c = ClassicalRegister(n, "c")
        i = 0
        for tpb_set in hamiltonian:
            circuits.append(copy.deepcopy(input_circuit))
            circuits_labels.append('tpb_circuit_' + str(i) + str(uuid.uuid4()))
            for j in range(n):
                # Measure X
                if tpb_set[0][1].v[j] == 0 and tpb_set[0][1].w[j] == 1:
                    circuits[i].h(q[j])
                # Measure Y
                elif tpb_set[0][1].v[j] == 1 and tpb_set[0][1].w[j] == 1:
                    circuits[i].s(q[j]).inverse()
                    circuits[i].h(q[j])
                circuits[i].measure(q[j], c[j])
            Q_program.add_circuit(circuits_labels[i], circuits[i])
            i += 1
        result = Q_program.execute(circuits_labels, device, shots=shots)
        for j, _ in enumerate(hamiltonian):
            for k, _ in enumerate(hamiltonian[j]):
                energy += hamiltonian[j][k][0] *\
                    measure_pauli_z(result.get_counts(
                        circuits_labels[j]), hamiltonian[j][k][1])

    return energy


def trial_circuit_ry(n, m, theta, entangler_map, meas_string=None,
                     measurement=True):
    """Creates a QuantumCircuit object ocnsisting in layers of
    parametrized single-qubit Y rotations and CZ two-qubit gates

    Args:
        n (int): number of qubits
        m (int): depth of the circuit
        theta (array[float]): angles that parametrize the Y rotations
        entangler_map (dict): CZ connectivity, e.g. {0: [1], 1: [2]}
        meas_string (str): measure a given Pauli operator at the end of the
            circuit
        measurement (bool): whether to measure the qubit (register "q")
            on classical bits (register "c")
    Returns:
        QuantumCircuit: A QuantumCircuit object
    """
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
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


def trial_circuit_ryrz(n, m, theta, entangler_map, meas_string=None,
                       measurement=True):
    """Creates a QuantumCircuit object consisting in layers of
    parametrized single-qubit Y and Z rotations and CZ two-qubit gates

    Args:
        n (int): number of qubits
        m (int): depth of the circuit
        theta (array[float]): angles that parametrize the Y and Z rotations
        entangler_map (dict): CZ connectivity, e.g. {0: [1], 1: [2]}
        meas_string (str): measure a given Pauli operator at the end of the
            circuit
        measurement (bool): whether to measure the qubit (register "q")
            on classical bits (register "c")
    Returns:
        QuantumCircuit: A QuantumCircuit object
    """
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")
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
            trial_circuit.ry(theta[n * i * 2 + 2 * j], q[j])
            trial_circuit.rz(theta[n * i * 2 + 2 * j + 1], q[j])
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
    """Creates a matrix operator out of a list of Paulis.

    Args:
        pauli_list (list): list of list [coeff,Pauli]
    Returns:
        numpy.matrix: A matrix representing pauli_list
    """
    Hamiltonian = 0
    for p in pauli_list:
        Hamiltonian += p[0] * p[1].to_matrix()
    return Hamiltonian


def Hamiltonian_from_file(file_name):
    """Creates a matrix operator out of a file with a list
    of Paulis.

    Args:
        file_name (str): a text file containing a list of Paulis and
        coefficients.
    Returns:
        list: A matrix representing pauli_list
    """
    with open(file_name, 'r+') as file:
        ham_array = file.readlines()
    ham_array = [x.strip() for x in ham_array]
    pauli_list = []
    for i in range(len(ham_array) // 2):
        pauli = label_to_pauli(ham_array[2 * i])
        Numb = float(ham_array[2 * i + 1])
        pauli_list.append([Numb, pauli])
    return pauli_list
