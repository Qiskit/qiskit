# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified QISKit usage."""
import logging
import warnings
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)

# Functions for importing qasm


def load_qasm_string(qasm_string, name=None):
    """Construct a quantum circuit from a qasm representation (string).

    Args:
        qasm_string (str): a string of qasm, or a filename containing qasm.
        name (str or None): the name of the quantum circuit after loading qasm
            text into it. If no name given, assign automatically.
    Returns:
        QuantumCircuit: circuit constructed from qasm.
    """
    warnings.warn('The load_qasm_string() function is deprecated and will be '
                  'removed in a future release. Instead use '
                  'QuantumCircuit.from_qasm_str().', DeprecationWarning)
    qc = QuantumCircuit.from_qasm_str(qasm_string)
    if name:
        qc.name = name
    return qc


def load_qasm_file(qasm_file, name=None):
    """Construct a quantum circuit from a qasm representation (file).

    Args:
        qasm_file (str): a string for the filename including its location.
        name (str or None): the name of the quantum circuit after
            loading qasm text into it. If no name is give the name is of
            the text file.
    Returns:
         QuantumCircuit: circuit constructed from qasm.
    """
    warnings.warn('The load_qasm_file() function is deprecated and will be '
                  'removed in a future release. Instead use '
                  'QuantumCircuit.from_qasm_file().', DeprecationWarning)
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    if name:
        qc.name = name
    return qc


def qobj_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits
    Returns:
        list: A list of QuantumCircuit objects from the qobj

    """
    if qobj.experiments:
        circuits = []
        for x in qobj.experiments:
            if hasattr(x.header, 'compiled_circuit_qasm'):
                circuits.append(
                    load_qasm_string(x.header.compiled_circuit_qasm))
        return circuits
    # TODO(mtreinish): add support for converting a qobj if the qasm isn't
    # embedded in the header
    return None
