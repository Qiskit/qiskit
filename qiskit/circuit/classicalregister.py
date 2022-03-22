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
Classical register reference object.
"""
import itertools

from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit


class Clbit(Bit):
    """Implement a classical bit."""

    def __init__(self, register=None, index=None):
        """Creates a classical bit.

        Args:
            register (ClassicalRegister): Optional. A classical register containing the bit.
            index (int): Optional. The index of the bit in its containing register.

        Raises:
            CircuitError: if the provided register is not a valid :class:`ClassicalRegister`
        """

        if register is None or isinstance(register, ClassicalRegister):
            super().__init__(register, index)
        else:
            raise CircuitError(
                "Clbit needs a ClassicalRegister and %s was provided" % type(register).__name__
            )


class ClassicalRegister(Register):
    """Implement a classical register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "c"
    bit_type = Clbit

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "creg %s[%d];" % (self.name, self.size)
