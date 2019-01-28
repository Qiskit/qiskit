# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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
This module contains the definition of creating and validating entangler map
based on the number of qubits.
"""

def get_entangler_map(map_type, num_qubits):
    """Utility method to get an entangler map among qubits

    Args:
        map_type (str): 'full' entangles each qubit with all the subsequent ones
                       'linear' entangles each qubit with the next
        num_qubits (int): Number of qubits for which the map is needed

    Returns:
        A map of qubit index to an array of indexes to which this should be entangled
    """
    ret = {}
    if num_qubits > 1:
        if map_type is None or map_type == 'full':
            ret = {i: [j for j in range(i, num_qubits) if j != i] for i in range(num_qubits-1)}
        elif map_type == 'linear':
            ret = {i: [i + 1] for i in range(num_qubits-1)}
    return ret

def validate_entangler_map(entangler_map, num_qubits, allow_double_entanglement=False):
    """Validates a user supplied entangler map and converts entries to ints

    Args:
        entangler_map (dict) : An entangler map, keys are source qubit index (int), value is array
                               of target qubit index(es) (int)
        num_qubits (int) : Number of qubits
        allow_double_entanglement: If we allow in list x entangled to y and vice-versa or not
    Returns:
        Validated/converted map
    """
    if not isinstance(entangler_map, dict):
        raise TypeError('Entangler map type dictionary expected')
    for k, v in entangler_map.items():
        if not isinstance(v, list):
            raise TypeError('Entangle index list expected but got {}'.format(type(v)))

    ret_map = {}
    for k, v in entangler_map.items():
        ret_map[int(k)] = [int(x) for x in v]

    for k, v in ret_map.items():
        if k < 0 or k >= num_qubits:
            raise ValueError('Qubit value {} invalid for {} qubits'.format(k, num_qubits))
        for i in v:
            if i < 0 or i >= num_qubits:
                raise ValueError('Qubit entangle target value {} invalid for {} qubits'.format(i, num_qubits))
            if allow_double_entanglement is False and i in ret_map and k in ret_map[i]:
                raise ValueError('Qubit {} and {} cross-listed'.format(i, k))
    return ret_map