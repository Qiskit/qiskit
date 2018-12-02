# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unitary gate.
"""
from ._instruction import Instruction
from ._qiskiterror import QiskitError


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, param, qargs, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters (will be converted to symbolic)
        qargs = list of pairs (QuantumRegister, index)
        circuit = QuantumCircuit containing this gate
        """
        self._is_multi_qubit = False
        self._qubit_coupling = [qarg[1] for qarg in qargs]
        self._is_multi_qubit = (len(qargs) > 1)

        super().__init__(name, param, qargs, [], circuit)

    def inverse(self):
        """Invert this gate."""
        raise QiskitError("inverse not implemented")

    def q_if(self, *qregs):
        """Add controls to this gate."""
        # pylint: disable=unused-argument
        raise QiskitError("control not implemented")
