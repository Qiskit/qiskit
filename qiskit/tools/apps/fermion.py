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
A set of functions to map fermionic Hamiltonians to qubit Hamiltonians.

References:
- E. Wigner and P. Jordan., Über das Paulische Äguivalenzverbot,
    Z. Phys., 47:631 (1928).
- S. Bravyi and A. Kitaev. Fermionic quantum computation,
    Ann. of Phys., 298(1):210–226 (2002).
- A. Tranter, S. Sofia, J. Seeley, M. Kaicher, J. McClean, R. Babbush,
    P. Coveney, F. Mintert, F. Wilhelm, and P. Love. The Bravyi–Kitaev
    transformation: Properties and applications. Int. Journal of Quantum
    Chemistry, 115(19):1431–1441 (2015).
- S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme,
    arXiv e-print arXiv:1701.08213 (2017).

"""
import numpy as np

from qiskit.tools.apps.optimization import Hamiltonian_from_file
from qiskit.tools.qi.pauli import Pauli, sgn_prod


def parity_set(j, n):
    """Computes the parity set of the j-th orbital in n modes

    Args:
        j (int) : the orbital index
        n (int) : the total number of modes
    Returns:
        numpy.array: Array of mode indexes
    """
    indexes = np.array([])
    if n % 2 == 0:
        if j < n / 2:
            indexes = np.append(indexes, parity_set(j, n / 2))
        else:
            indexes = np.append(indexes, np.append(
                parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
    return indexes


def update_set(j, n):
    """Computes the update set of the j-th orbital in n modes

    Args:
        j (int) : the orbital index
        n (int) : the total number of modes
    Returns:
        numpy.array: Array of mode indexes
    """
    indexes = np.array([])
    if n % 2 == 0:
        if j < n / 2:
            indexes = np.append(indexes, np.append(
                n - 1, update_set(j, n / 2)))
        else:
            indexes = np.append(indexes, update_set(j - n / 2, n / 2) + n / 2)
    return indexes


def flip_set(j, n):
    """Computes the flip set of the j-th orbital in n modes

    Args:
        j (int) : the orbital index
        n (int) : the total number of modes
    Returns:
        numpy.array: Array of mode indexes
    """
    indexes = np.array([])
    if n % 2 == 0:
        if j < n / 2:
            indexes = np.append(indexes, flip_set(j, n / 2))
        elif j >= n / 2 and j < n - 1:
            indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
        else:
            indexes = np.append(np.append(indexes, flip_set(
                j - n / 2, n / 2) + n / 2), n / 2 - 1)
    return indexes


def pauli_term_append(pauli_term, pauli_list, threshold):
    """Appends a Pauli term to a Pauli list

    If pauli_term is already present in the list adjusts the coefficient
    of the existing pauli. If the new coefficient is less than
    threshold the pauli term is deleted from the list

    Args:
        pauli_term (list): list of [coeff, pauli]
        pauli_list (list): a list of pauli_terms
        threshold (float): simplification threshold
    Returns:
        list: an updated pauli_list
    """
    found = False
    if np.absolute(pauli_term[0]) > threshold:
        if pauli_list:   # if the list is not empty
            for i, _ in enumerate(pauli_list):
                # check if the new pauli belongs to the list
                if pauli_list[i][1] == pauli_term[1]:
                    # if found renormalize the coefficient of existent pauli
                    pauli_list[i][0] += pauli_term[0]
                    # remove the element if coeff. value is now less than
                    # threshold
                    if np.absolute(pauli_list[i][0]) < threshold:
                        del pauli_list[i]
                    found = True
                    break
            if found is False:       # if not found add the new pauli
                pauli_list.append(pauli_term)
        else:
            # if list is empty add the new pauli
            pauli_list.append(pauli_term)
    return pauli_list


def fermionic_maps(h1, h2, map_type, out_file=None, threshold=0.000000000001):
    """Creates a list of Paulis with coefficients from fermionic one and
    two-body operator.

    Args:
        h1 (list): second-quantized fermionic one-body operator
        h2 (list): second-quantized fermionic two-body operator
        map_type (str): "JORDAN_WIGNER", "PARITY", "BINARY_TREE"
        out_file (str): name of the optional file to write the Pauli list on
        threshold (float): threshold for Pauli simplification
    Returns:
        list: A list of Paulis with coefficients
    """
    # pylint: disable=invalid-name

    ####################################################################
    # ###########   DEFINING MAPPED FERMIONIC OPERATORS    #############
    ####################################################################

    pauli_list = []
    n = len(h1)  # number of fermionic modes / qubits
    a = []
    if map_type == 'JORDAN_WIGNER':
        for i in range(n):
            xv = np.append(np.append(np.ones(i), 0), np.zeros(n - i - 1))
            xw = np.append(np.append(np.zeros(i), 1), np.zeros(n - i - 1))
            yv = np.append(np.append(np.ones(i), 1), np.zeros(n - i - 1))
            yw = np.append(np.append(np.zeros(i), 1), np.zeros(n - i - 1))
            # defines the two mapped Pauli components of a_i and a_i^\dag,
            # according to a_i -> (a[i][0]+i*a[i][1])/2,
            # a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((Pauli(xv, xw), Pauli(yv, yw)))
    if map_type == 'PARITY':
        for i in range(n):
            if i > 1:
                Xv = np.append(np.append(np.zeros(i - 1),
                                         [1, 0]), np.zeros(n - i - 1))
                Xw = np.append(np.append(np.zeros(i - 1),
                                         [0, 1]), np.ones(n - i - 1))
                Yv = np.append(np.append(np.zeros(i - 1),
                                         [0, 1]), np.zeros(n - i - 1))
                Yw = np.append(np.append(np.zeros(i - 1),
                                         [0, 1]), np.ones(n - i - 1))
            elif i > 0:
                Xv = np.append((1, 0), np.zeros(n - i - 1))
                Xw = np.append([0, 1], np.ones(n - i - 1))
                Yv = np.append([0, 1], np.zeros(n - i - 1))
                Yw = np.append([0, 1], np.ones(n - i - 1))
            else:
                Xv = np.append(0, np.zeros(n - i - 1))
                Xw = np.append(1, np.ones(n - i - 1))
                Yv = np.append(1, np.zeros(n - i - 1))
                Yw = np.append(1, np.ones(n - i - 1))
            # defines the two mapped Pauli components of a_i and a_i^\dag,
            # according to a_i -> (a[i][0]+i*a[i][1])/2,
            # a_i^\dag -> (a_[i][0]-i*a[i][1])/2
            a.append((Pauli(Xv, Xw), Pauli(Yv, Yw)))
    if map_type == 'BINARY_TREE':
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        while n > np.power(2, bin_sup):
            bin_sup += 1
        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
        update_sets = []
        update_pauli = []

        parity_sets = []
        parity_pauli = []

        flip_sets = []

        remainder_sets = []
        remainder_pauli = []
        for j in range(n):

            update_sets.append(update_set(j, np.power(2, bin_sup)))
            update_sets[j] = update_sets[j][update_sets[j] < n]

            parity_sets.append(parity_set(j, np.power(2, bin_sup)))
            parity_sets[j] = parity_sets[j][parity_sets[j] < n]

            flip_sets.append(flip_set(j, np.power(2, bin_sup)))
            flip_sets[j] = flip_sets[j][flip_sets[j] < n]

            remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

            update_pauli.append(Pauli(np.zeros(n), np.zeros(n)))
            parity_pauli.append(Pauli(np.zeros(n), np.zeros(n)))
            remainder_pauli.append(Pauli(np.zeros(n), np.zeros(n)))
            for k in range(n):
                if np.in1d(k, update_sets[j]):
                    update_pauli[j].w[k] = 1
                if np.in1d(k, parity_sets[j]):
                    parity_pauli[j].v[k] = 1
                if np.in1d(k, remainder_sets[j]):
                    remainder_pauli[j].v[k] = 1

            x_j = Pauli(np.zeros(n), np.zeros(n))
            x_j.w[j] = 1
            y_j = Pauli(np.zeros(n), np.zeros(n))
            y_j.v[j] = 1
            y_j.w[j] = 1
            # defines the two mapped Pauli components of a_i and a_i^\dag,
            # according to a_i -> (a[i][0]+i*a[i][1])/2, a_i^\dag ->
            # (a_[i][0]-i*a[i][1])/2
            a.append((update_pauli[j] * x_j * parity_pauli[j],
                      update_pauli[j] * y_j * remainder_pauli[j]))

    ####################################################################
    # ###########    BUILDING THE MAPPED HAMILTONIAN     ###############
    ####################################################################

    # ######################    One-body    #############################
    for i in range(n):
        for j in range(n):
            if h1[i, j] != 0:
                for alpha in range(2):
                    for beta in range(2):
                        pauli_prod = sgn_prod(a[i][alpha], a[j][beta])
                        pauli_term = [h1[i, j] * 1 / 4 * pauli_prod[1] *
                                      np.power(-1j, alpha) *
                                      np.power(1j, beta),
                                      pauli_prod[0]]
                        pauli_list = pauli_term_append(
                            pauli_term, pauli_list, threshold)

    # ######################    Two-body    ############################
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    if h2[i, j, k, m] != 0:
                        for alpha in range(2):
                            for beta in range(2):
                                for gamma in range(2):
                                    for delta in range(2):
                                        # Note: chemists' notation for the
                                        # labeling,
                                        # h2(i,j,k,m) adag_i adag_k a_m a_j
                                        pauli_prod_1 = sgn_prod(
                                            a[i][alpha], a[k][beta])
                                        pauli_prod_2 = sgn_prod(
                                            pauli_prod_1[0], a[m][gamma])
                                        pauli_prod_3 = sgn_prod(
                                            pauli_prod_2[0], a[j][delta])

                                        phase1 = pauli_prod_1[1] *\
                                            pauli_prod_2[1] * pauli_prod_3[1]
                                        phase2 = np.power(-1j, alpha + beta) *\
                                            np.power(1j, gamma + delta)

                                        pauli_term = [
                                            h2[i, j, k, m] * 1 / 16 * phase1 *
                                            phase2, pauli_prod_3[0]]

                                        pauli_list = pauli_term_append(
                                            pauli_term, pauli_list, threshold)

    ####################################################################
    # ################          WRITE TO FILE         ##################
    ####################################################################

    if out_file is not None:
        out_stream = open(out_file, 'w')
        for pauli_term in pauli_list:
            out_stream.write(pauli_term[1].to_label() + '\n')
            out_stream.write('%.15f' % pauli_term[0].real + '\n')
        out_stream.close()
    return pauli_list


def two_qubit_reduction(ham_in, m, out_file=None, threshold=0.000000000001):
    """
    Eliminates the central and last qubit in a list of Pauli that has
    diagonal operators (Z,I) at those positions.abs

    It can be used to taper two qubits in parity and binary-tree mapped
    fermionic Hamiltonians when the spin orbitals are ordered in two spin
    sectors, according to the number of particles in the system.

    Args:
        ham_in (list): a list of Paulis representing the mapped fermionic
            Hamiltonian
        m (int): number of fermionic particles
        out_file (string or None): name of the optional file to write the Pauli
            list on
        threshold (float): threshold for Pauli simplification
    Returns:
        list: A tapered Hamiltonian in the form of list of Paulis with
            coefficients
    """
    ham_out = []
    if m % 4 == 0:
        par_1 = 1
        par_2 = 1
    elif m % 4 == 1:
        par_1 = -1
        par_2 = -1    # could be also +1, +1/-1 are  spin-parity sectors
    elif m % 4 == 2:
        par_1 = 1
        par_2 = -1
    else:
        par_1 = -1
        par_2 = -1    # could be also +1, +1/-1 are  spin-parity sectors
    if isinstance(ham_in, str):
        # conversion from Hamiltonian text file to pauli_list
        ham_in = Hamiltonian_from_file(ham_in)
    # number of qubits
    n = len(ham_in[0][1].v)
    for pauli_term in ham_in:  # loop over Pauli terms
        coeff_out = pauli_term[0]
        # Z operator encountered at qubit n/2-1
        if pauli_term[1].v[n // 2 -
                           1] == 1 and pauli_term[1].w[n // 2 - 1] == 0:
            coeff_out = par_2 * coeff_out
        # Z operator encountered at qubit n-1
        if pauli_term[1].v[n - 1] == 1 and pauli_term[1].w[n - 1] == 0:
            coeff_out = par_1 * coeff_out
        v_temp = []
        w_temp = []
        for j in range(n):
            if j != n // 2 - 1 and j != n - 1:
                v_temp.append(pauli_term[1].v[j])
                w_temp.append(pauli_term[1].w[j])
        pauli_term_out = [coeff_out, Pauli(v_temp, w_temp)]
        ham_out = pauli_term_append(pauli_term_out, ham_out, threshold)

    ####################################################################
    # ################          WRITE TO FILE         ##################
    ####################################################################

    if out_file is not None:
        out_stream = open(out_file, 'w')
        for pauli_term in ham_out:
            out_stream.write(pauli_term[1].to_label() + '\n')
            out_stream.write('%.15f' % pauli_term[0].real + '\n')
        out_stream.close()
    return ham_out
