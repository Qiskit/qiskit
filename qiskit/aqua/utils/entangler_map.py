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
"""
This module contains the definition of creating and validating entangler map
based on the number of qubits.
"""


def get_entangler_map(map_type, num_qubits):
    """Utility method to get an entangler map among qubits.

    Args:
        map_type (str): 'full' entangles each qubit with all the subsequent ones
                       'linear' entangles each qubit with the next
        num_qubits (int): Number of qubits for which the map is needed

    Returns:
        A map of qubit index to an array of indexes to which this should be entangled

    Raises:
        ValueError: if map_type is not valid.
    """
    ret = []
    if num_qubits > 1:
        if map_type == 'full':
            ret = [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif map_type == 'linear':
            ret = [[i, i + 1] for i in range(num_qubits - 1)]
        else:
            raise ValueError("map_type only supports 'full' or 'linear' type.")
    return ret


def validate_entangler_map(entangler_map, num_qubits, allow_double_entanglement=False):
    """Validate a user supplied entangler map and converts entries to ints.

    Args:
        entangler_map (list[list]) : An entangler map, keys are source qubit index (int), value is array
                               of target qubit index(es) (int)
        num_qubits (int) : Number of qubits
        allow_double_entanglement: If we allow in two qubits can be entangled each other

    Returns:
        Validated/converted map

    Raises:
        TypeError: entangler map is not list type or list of list
        ValueError: the index of entangler map is out of range
        ValueError: the qubits are cross-entangled.

    """

    if isinstance(entangler_map, dict):
        raise TypeError("The type of entangler map is changed to list of list.")

    if not isinstance(entangler_map, list):
        raise TypeError("Entangler map type 'list' expected")

    for src_to_targ in entangler_map:
        if not isinstance(src_to_targ, list):
            raise TypeError('Entangle index list expected but got {}'.format(type(src_to_targ)))

    ret_map = []
    ret_map = [[int(src), int(targ)] for src, targ in entangler_map]

    for src, targ in ret_map:
        if src < 0 or src >= num_qubits:
            raise ValueError('Qubit entangle source value {} invalid for {} qubits'.format(src, num_qubits))
        if targ < 0 or targ >= num_qubits:
            raise ValueError('Qubit entangle target value {} invalid for {} qubits'.format(targ, num_qubits))
        if not allow_double_entanglement and [targ, src] in ret_map:
            raise ValueError('Qubit {} and {} cross-entangled.'.format(src, targ))

    return ret_map
