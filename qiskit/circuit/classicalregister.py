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

# pylint: disable=bad-docstring-quotes

"""
Classical register reference object.
"""
import itertools

from qiskit.utils.deprecation import deprecate_func
from .register import Register
from .bit import Bit


class Clbit(Bit):
    """Implement a classical bit."""

    pass


class ClassicalRegister(Register):
    """Implement a classical register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "c"
    bit_type = Clbit

    @deprecate_func(
        additional_msg=(
            "Correct exporting to OpenQASM 2 is the responsibility of a larger exporter; it cannot "
            "safely be done on an object-by-object basis without context. No replacement will be "
            "provided, because the premise is wrong."
        ),
        since="0.23.0",
        package_name="qiskit-terra",
    )
    def qasm(self):
        """Return OPENQASM string for this register."""
        return "creg %s[%d];" % (self.name, self.size)
