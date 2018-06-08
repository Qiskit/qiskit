# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Gates tools."""

from qiskit import InstructionSet
from qiskit import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


def attach_gate(element, quantum_register, gate, gate_class):
    """Attach a gate."""
    if isinstance(quantum_register, QuantumRegister):
        gs = InstructionSet()
        for _ in range(quantum_register.size):
            gs.add(gate)
        return gs

    element._check_qubit(quantum_register)
    return element._attach(gate_class)
