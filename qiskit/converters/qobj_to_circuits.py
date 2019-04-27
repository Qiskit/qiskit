# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting qobj to a list of circuits"""

import warnings

from qiskit.compiler import disassembler


def qobj_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits
    Returns:
        list: A list of QuantumCircuit objects from the qobj

    """
    warnings.warn('qiskit.converters.qobj_to_circuit() is deprecated and will '
                  'be removed in Qiskit Terra 0.9. Please use '
                  'qiskit.compiler.disassemble_circuits() to convert a qobj '
                  'to list of circuits.', DeprecationWarning)
    return disassembler._experiments_to_circuits(qobj)
