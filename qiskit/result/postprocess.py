# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Post-processing of raw result."""

import warnings
import numpy as np
from qiskit import QiskitError, QuantumCircuit
from qiskit.validation.base import BaseModel, bind_schema
from .models import ResultSchema


def _hex_to_bin(self, hexstring):
    """Convert hexadecimal readouts (memory) to binary readouts."""
    return str(bin(int(hexstring, 16)))[2:]

def _pad_zeros(self, bitstring, memory_slots):
    """If the bitstring is truncated, pad extra zeros to make its
    length equal to memory_slots"""
    return format(int(bitstring, 2), '0{}b'.format(memory_slots))

def _histogram(self, outcomes):
    """Build histogram from measurement outcomes of each shot."""
    counts = dict(Counter(outcomes))
    return counts

def _little_endian(self, bitstring, clbit_labels, creg_sizes):
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

def _separate_bitstring(self, bitstring, creg_sizes):
    """Separate a bitstring according to the registers defined in the result header."""
    substrings = []
    running_index = 0
    for reg, size in creg_sizes:
        substrings.append(bitstring[running_index: running_index + size])
        running_index += size
    return ' '.join(substrings)

def _format_resultstring(self, resultstring, exp_result_header):
    """
    Convert from hex to binary, make little endian, and insert space between registers.
    """
    creg_sizes = exp_result_header.get('creg_sizes')
    clbit_labels = exp_result_header.get('clbit_labels')
    memory_slots = exp_result_header.get('memory_slots')
    if resultstring.startswith('0x'):
        resultstring = self._hex_to_bin(resultstring)
    if memory_slots:
        resultstring = self._pad_zeros(resultstring, memory_slots)
    if clbit_labels and creg_sizes:
        resultstring = self._little_endian(resultstring, clbit_labels, creg_sizes)
    if creg_sizes:
        resultstring = self._separate_bitstring(resultstring, creg_sizes)
    return resultstring

def format_readout(self, exp_result):
    """Format a single experiment result coming from backend to present
    to the Qiskit user.

    Histograms "counts" are created from "memory" data (if the backend has
    not already created them)
    
    The hexadecimals are expanded to bitstrings
    
    The order is made little endian (LSB on the right)
    
    Spaces are inserted at register divisions.

    Args:
        exp_result (ExperimentResult): result of a single experiment
    """
    if 'memory' in exp_result.data and 'counts' not in exp_result.data:
        exp_result.data['counts'] = self._histogram(exp_result.data['memory'])
    memory_list = []
    for element in exp_result.data.get('memory', []):
        element = self._format_resultstring(element, exp_result.header)
        memory_list.append(element)
    exp_result.data['memory'] = memory_list
    counts_dict = {}
    for key, val in exp_result.data.get('counts', {}).items():
        key = self._format_resultstring(key, exp_result.header)
        counts_dict[key] = val
    exp_result.data['counts'] = counts_dict

def format_statevector(self, exp_result):
    """Format statevector coming from the backend to present to the Qiskit user.

    Args:
        exp_result (ExperimentResult): result of a single experiment
    """
