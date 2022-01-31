# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum register reference object.
"""
import itertools

from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit


class Qubit(Bit):
    """Implement a quantum bit."""

    def __init__(self, register=None, index=None):
        """Creates a qubit.

        Args:
            register (QuantumRegister): Optional. A quantum register containing the bit.
            index (int): Optional. The index of the bit in its containing register.

        Raises:
            CircuitError: if the provided register is not a valid :class:`QuantumRegister`
        """

        if register is None or isinstance(register, QuantumRegister):
            super().__init__(register, index)
        else:
            raise CircuitError(
                "Qubit needs a QuantumRegister and %s was provided" % type(register).__name__
            )


class QuantumRegister(Register):
    """Implement a quantum register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "q"
    bit_type = Qubit

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)


class AncillaQubit(Qubit):
    """A qubit used as ancillary qubit."""

    pass


class AncillaRegister(QuantumRegister):
    """Implement an ancilla register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "a"
    bit_type = AncillaQubit
