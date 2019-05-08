# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting qobj to a list of circuits"""

import warnings

from qiskit.assembler import disassemble


def qobj_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits
    Returns:
        list: A list of QuantumCircuit objects from the qobj

    """
    warnings.warn('qiskit.converters.qobj_to_circuit() is deprecated and will '
                  'be removed in Qiskit Terra 0.9. Please use '
                  'qiskit.assembler.disassemble() to convert a qobj '
                  'to list of circuits.', DeprecationWarning)

    variables = disassemble(qobj)
    return variables[0]
