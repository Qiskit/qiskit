# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Post-processing of raw result."""

import numpy as np


def _hex_to_bin(hexstring):
    """Convert hexadecimal readouts (memory) to binary readouts."""
    return str(bin(int(hexstring, 16)))[2:]


def _pad_zeros(bitstring, memory_slots):
    """If the bitstring is truncated, pad extra zeros to make its
    length equal to memory_slots"""
    return format(int(bitstring, 2), '0{}b'.format(memory_slots))


def _separate_bitstring(bitstring, creg_sizes):
    """Separate a bitstring according to the registers defined in the result header."""
    substrings = []
    running_index = 0
    for _, size in reversed(creg_sizes):
        substrings.append(bitstring[running_index: running_index + size])
        running_index += size
    return ' '.join(substrings)


def format_memory(memory, header):
    """
    Format a single bitstring (memory) from a single shot experiment.

    - The hexadecimals are expanded to bitstrings

    - Spaces are inserted at register divisions.

    Args:
        memory (str): result of a single experiment.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        dict: a formatted memory
    """
    creg_sizes = header.get('creg_sizes')
    memory_slots = header.get('memory_slots')
    if memory.startswith('0x'):
        memory = _hex_to_bin(memory)
    if memory_slots:
        memory = _pad_zeros(memory, memory_slots)
    if creg_sizes:
        memory = _separate_bitstring(memory, creg_sizes)
    return memory


def format_counts(counts, header):
    """Format a single experiment result coming from backend to present
    to the Qiskit user.

    Args:
        counts (dict): counts histogram of multiple shots
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        dict: a formatted counts
    """
    counts_dict = {}
    for key, val in counts.items():
        key = format_memory(key, header)
        counts_dict[key] = val
    return counts_dict


def format_statevector(vec, decimals=None):
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
    if decimals:
        vec_complex = np.around(vec_complex, decimals=decimals)
    return vec_complex


def format_unitary(mat, decimals=None):
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
