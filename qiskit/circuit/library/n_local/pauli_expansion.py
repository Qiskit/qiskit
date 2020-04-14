# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from typing import Optional, Callable, List, Union
import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators import evolution_instruction

from .data_mapping import self_product
from .n_local import NLocal


logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class PauliExpansion(NLocal):
    r"""The Pauli Expansion feature map.

    Refer to https://arxiv.org/abs/1804.11326 for details.
    The Pauli Expansion feature map transforms data :math:`\vec{x} \in \mathbb{R}^n`
    according to the following equation, and then duplicate the same circuit with depth
    :math:`d` times, where :math:`d` is the depth of the circuit:

    :math:`U_{\Phi(\vec{x})}=\exp\left(i\sum_{S\subseteq [n]}
    \phi_S(\vec{x})\prod_{i\in S} P_i\right)`

    where :math:`S \in \{\binom{n}{k}\ combinations,\ k = 1,... n \}, \phi_S(\vec{x}) = x_i` if
    :math:`k=1`, otherwise :math:`\phi_S(\vec{x}) = \prod_S(\pi - x_j)`, where :math:`j \in S`, and
    :math:`P_i \in \{ I, X, Y, Z \}`

    Please refer to :class:`FirstOrderExpansion` for the case
    :math:`k = 1`, :math:`P_0 = Z`
    and to :class:`SecondOrderExpansion` for the case
    :math:`k = 2`, :math:`P_0 = Z\ and\ P_1 P_0 = ZZ`.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 paulis: Optional[List[str]] = None,
                 data_map_func: Callable[[np.ndarray], float] = self_product,
                 insert_barriers: bool = False) -> None:
        """
        Args:
            feature_dimension: Number of features.
            depth: The number of repeated circuits. Defaults to 2, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
            paulis: A list of strings for to-be-used paulis. Defaults to None.
                If None, ['Z', 'ZZ'] will be used.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
        """
        paulis = paulis if paulis is not None else ['Z', 'ZZ']

        super().__init__(insert_barriers=insert_barriers, overwrite_block_parameters=False,
                         entanglement=entanglement)

        self._num_qubits = feature_dimension
        self._entanglement = entanglement
        self._pauli_strings = self._build_subset_paulis_string(paulis)
        self._data_map_func = data_map_func

        # define a hadamard layer for convenience
        hadamards = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            hadamards.h(i)

        # set the parameters
        x = ParameterVector('x', length=feature_dimension)

        # iterate over the layers
        for _ in range(depth):
            self += hadamards
            for pauli in self._pauli_strings:
                coeff = self._data_map_func(self._extract_data_for_rotation(pauli, x))
                p = Pauli.from_label(pauli)
                inst = evolution_instruction([[1, p]], coeff, 1)
                self.append(inst)

    @property
    def feature_dimension(self) -> int:
        """Returns the feature dimension (which is equal to the number of qubits).

        Returns:
            The feature dimension of this feature map.
        """
        return self.num_qubits

    def _build_subset_paulis_string(self, paulis):
        # fill out the paulis to the number of qubits
        temp_paulis = []
        for pauli in paulis:
            len_pauli = len(pauli)
            for possible_pauli_idx in itertools.combinations(range(self.num_qubits), len_pauli):
                string_temp = ['I'] * self.num_qubits
                for idx, _ in enumerate(possible_pauli_idx):
                    string_temp[-possible_pauli_idx[idx] - 1] = pauli[-idx - 1]
                temp_paulis.append(''.join(string_temp))
        # clean up string that can not be entangled.
        final_paulis = []
        for pauli in temp_paulis:
            where_z = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
            if len(where_z) == 1:
                final_paulis.append(pauli)
            else:
                is_valid = True
                for control, target in itertools.combinations(where_z, 2):
                    if [control, target] not in self.get_entangler_map(2, self.num_qubits,
                                                                       self.entanglement):
                        is_valid = False
                        break
                if is_valid:
                    final_paulis.append(pauli)
                else:
                    logger.warning("Due to the limited entangler_map, %s is skipped.", pauli)

        logger.info("Pauli terms include: %s", final_paulis)
        return final_paulis

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        x = np.asarray(x)
        return x[where_non_i]
