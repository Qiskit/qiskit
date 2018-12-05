# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np

"""Post-processing of raw result."""

def _hex_to_bin(hexstring):
    """Convert hexadecimal readouts (memory) to binary readouts."""
    return str(bin(int(hexstring, 16)))[2:]


def _pad_zeros(bitstring, memory_slots):
    """If the bitstring is truncated, pad extra zeros to make its
    length equal to memory_slots"""
    return format(int(bitstring, 2), '0{}b'.format(memory_slots))


def _little_endian(bitstring, clbit_labels, creg_sizes):
    """
    Reorder the bitstring to little endian (Least Significant Bit last, and
    Least Significant Register last).
    """
    # backend already reports the full memory as little_endian.
    # reverse the original bitstring to get bitstring[i] to correspond to clbit_label[i]
    bitstring = bitstring[::-1]
    # registers appearing first in the QuantumCircuit declaration are more significant
    register_significance = lambda r: [i for i, reg in enumerate(creg_sizes) if reg[0] == r][0]
    # higher indices are more significant
    index_significance = lambda i: -i
    # order: more significant register first, more significant bit first
    key = lambda position: (register_significance(clbit_labels[position][0]),
                            index_significance(clbit_labels[position][1]))
    return ''.join([bitstring[i] for i in sorted(range(len(bitstring)), key=key)])


def _separate_bitstring(bitstring, creg_sizes):
    """Separate a bitstring according to the registers defined in the result header."""
    substrings = []
    running_index = 0
    for reg, size in reversed(creg_sizes):
        substrings.append(bitstring[running_index: running_index + size])
        running_index += size
    return ' '.join(substrings)


def format_memory(memory, header):
    """
    Format a single bitstring (memory) from a single shot experiment.

    - The hexadecimals are expanded to bitstrings
    
    - The order is made little endian (LSB on the right)
    
    - Spaces are inserted at register divisions.
    
    Args:
        memory (str): result of a single experiment.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        dict: a formatted memory
    """
    creg_sizes = header.get('creg_sizes')
    clbit_labels = header.get('clbit_labels')
    memory_slots = header.get('memory_slots')
    if memory.startswith('0x'):
        memory = _hex_to_bin(memory)
    if memory_slots:
        memory = _pad_zeros(memory, memory_slots)
    #if clbit_labels and creg_sizes:
        #memory = _little_endian(memory, clbit_labels, creg_sizes)
    if creg_sizes:
        memory = _separate_bitstring(memory, creg_sizes)
    return memory


def format_counts(counts, header):
    """Format a single experiment result coming from backend to present
    to the Qiskit user.

    Args:
        counts (dict{str: int}): counts histogram of multiple shots
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        dict: a formatted counts
    """
    """
    if 'memory' in exp_result.data and 'counts' not in exp_result.data:
        exp_result.data['counts'] = _histogram(exp_result.data['memory'])
    memory_list = []
    for element in exp_result.data.get('memory', []):
        element = _format_resultstring(element, exp_result.header)
        memory_list.append(element)
    exp_result.data['memory'] = memory_list
    """
    counts_dict = {}
    for key, val in counts.items():
        key = format_memory(key, header)
        counts_dict[key] = val
    return counts_dict


def format_statevector(vec, decimals=8):
    """Format statevector coming from the backend to present to the Qiskit user.

    Args:
        vec (list): a list of [re, im] complex numbers.
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[complex]: a list of python complex numbers.
    """
    num_basis = len(vec)
    vec_complex = np.zeros(num_basis, dtype=complex)
    for i in range(num_basis):
        vec_complex[i] = vec[i][0] + 1j * vec[i][1]
    if not decimals:
        return vec_complex
    else:
        return np.around(vec_complex, decimals=decimals)


def format_unitary(mat, decimals=8):
    """Format unitary coming from the backend to present to the Qiskit user.

    Args:
        mat (list[list]): a list of list of [re, im] complex numbers
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[list[complex]]: a matrix of complex numbers
    """
    num_basis = len(mat)
    mat_complex = np.zeros((num_basis, num_basis), dtype=complex)
    for i, vec in enumerate(mat):
        mat_complex[i] = format_statevector(vec, decimals)
    return mat_complex
