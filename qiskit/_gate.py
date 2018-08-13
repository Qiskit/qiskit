# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unitary gate.
"""
from ._instruction import Instruction
from ._quantumregister import QuantumRegister
from ._qiskiterror import QISKitError


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, param, args, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters (will be converted to symbolic)
        args = list of pairs (Register, index)
        circuit = QuantumCircuit or CompositeGate containing this gate
        """
        self._is_multi_qubit = False
        self._qubit_coupling = [arg[1] for arg in args]
        self._is_multi_qubit = (len(args) > 1)
        for argument in args:
            if not isinstance(argument[0], QuantumRegister):
                raise QISKitError("argument not (QuantumRegister, int) "
                                  + "tuple")

        super().__init__(name, param, args, circuit)

    def inverse(self):
        """Invert this gate."""
        raise QISKitError("inverse not implemented")

    def q_if(self, *qregs):
        """Add controls to this gate."""
        # pylint: disable=unused-argument
        raise QISKitError("control not implemented")
