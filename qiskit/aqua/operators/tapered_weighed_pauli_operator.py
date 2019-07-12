# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua import AquaError
from qiskit.aqua.operators.weighted_pauli_operator import WeightedPauliOperator


logger = logging.getLogger(__name__)


class TaperedWeightedPauliOperator(WeightedPauliOperator):

    def __init__(self, paulis, symmetries, cliffords, sq_list, tapering_values, basis=None, atol=1e-12):
        super().__init__(paulis, basis, atol)
        self._symmetries = symmetries
        self._cliffords = cliffords
        self._sq_list = sq_list
        self._tapering_values = tapering_values

    @property
    def symmetries(self):
        return self._symmetries

    @property
    def cliffords(self):
        return self._cliffords

    @property
    def sq_list(self):
        return self._sq_list

    @property
    def tapering_values(self):
        return self._tapering_values

    @classmethod
    def qubit_tapering(cls, operator, symmetries, cliffords, sq_list, tapering_values):
        """
        Builds an Operator which has a number of qubits tapered off,
        based on a block-diagonal Operator built using a list of cliffords.
        The block-diagonal subspace is an input parameter, set through the list
        tapering_values, which takes values +/- 1.

        Args:
            operator (WeightedPauliOperator): the target operator to be tapered
            symmetries ([Pauli]): the list of Pauli objects representing the Z_2 symmetries
            cliffords ([WeightedPauliOperator]): list of unitary Clifford transformation
            sq_list ([int]): position of the single-qubit operators that anticommute
                             with the cliffords
            tapering_values ([int]): array of +/- 1 used to select the subspace. Length
                                     has to be equal to the length of cliffords and sq_list
        Returns:
            WeightedPauliOperator : the tapered operator, or empty operator if the `operator` is empty.
        """
        if len(cliffords) == 0 or len(sq_list) == 0 or len(tapering_values) == 0:
            logger.warning("Cliffords, single qubit list and tapering values cannot be empty.\n"
                           "Return the original operator instead.")
            return operator

        if len(cliffords) != len(sq_list):
            logger.warning("Number of Clifford unitaries has to be the same as length of single"
                           "qubit list and tapering values.\n"
                           "Return the original operator instead.")
            return operator
        if len(sq_list) != len(tapering_values):
            logger.warning("Number of Clifford unitaries has to be the same as length of single"
                           "qubit list and tapering values.\n"
                           "Return the original operator instead.")
            return operator

        if operator.is_empty():
            logger.warning("The operator is empty, return the empty operator directly.")
            return operator

        for clifford in cliffords:
            operator = clifford * operator * clifford

        operator_out = []
        for pauli_term in operator.paulis:
            coeff_out = pauli_term[0]
            for idx, qubit_idx in enumerate(sq_list):
                if not (not pauli_term[1].z[qubit_idx] and not pauli_term[1].x[qubit_idx]):
                    coeff_out = tapering_values[idx] * coeff_out
            z_temp = np.delete(pauli_term[1].z.copy(), np.asarray(sq_list))
            x_temp = np.delete(pauli_term[1].x.copy(), np.asarray(sq_list))
            pauli_term_out = [coeff_out, Pauli(z_temp, x_temp)]
            operator_out.extend([pauli_term_out])

        return cls(operator_out, symmetries, cliffords, sq_list, tapering_values)

    @classmethod
    def two_qubit_reduction(cls, operator, num_particles):
        """
        Eliminates the central and last qubit in a list of Pauli that has
        diagonal operators (Z,I) at those positions

        Chemistry specific method:
        It can be used to taper two qubits in parity and binary-tree mapped
        fermionic Hamiltonians when the spin orbitals are ordered in two spin
        sectors, (block spin order) according to the number of particles in the system.

        Args:
            operator (WeightedPauliOperator): the operator
            num_particles (list, int): number of particles, if it is a list, the first number is alpha
                                        and the second number if beta.

        Returns:
            Operator: a new operator whose qubit number is reduced by 2.

        """
        if operator._paulis is None or operator._paulis == []:
            return operator

        if isinstance(num_particles, list):
            num_alpha = num_particles[0]
            num_beta = num_particles[1]
        else:
            num_alpha = num_particles // 2
            num_beta = num_particles // 2

        par_1 = 1 if (num_alpha + num_beta) % 2 == 0 else -1
        par_2 = 1 if num_alpha % 2 == 0 else -1
        tapering_values = [par_2, par_1]

        num_qubits = operator.num_qubits
        last_idx = num_qubits - 1
        mid_idx = num_qubits // 2 - 1
        sq_list = [mid_idx, last_idx]

        # build symmetries, sq_paulis, cliffords:
        symmetries, sq_paulis, cliffords = [], [], []
        for idx in sq_list:
            pauli_str = ['I'] * num_qubits

            pauli_str[idx] = 'Z'
            z_sym = Pauli.from_label(''.join(pauli_str)[::-1])
            symmetries.append(z_sym)

            pauli_str[idx] = 'X'
            sq_pauli = Pauli.from_label(''.join(pauli_str)[::-1])
            sq_paulis.append(sq_pauli)

            clifford = WeightedPauliOperator(paulis=[[1. / np.sqrt(2), z_sym], [1. / np.sqrt(2), sq_pauli]])
            cliffords.append(clifford)

        return cls.qubit_tapering(operator, symmetries, cliffords, sq_list, tapering_values)

    def tapering_consistently(self, operator):
        """
        Tapering the `operator` with the same manner of how this tapered operator is created. i.e., using the same
        cliffords and tapering values.

        Args:
            operator (WeightedPauliOperator): the to-be-tapered operator

        Returns:
            TaperedWeightedPauliOperator: the tapered operator
        """
        if operator.is_empty():
            raise AquaError("Can not taper an empty operator.")

        for symmetry in self._symmetries:
            if not operator.is_commute(symmetry):
                raise AquaError("The given operator does not commute with the symmetry, can not taper it.")

        return self.qubit_tapering(operator, self._symmetries, self._cliffords, self._sq_list, self._tapering_values)
