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
Exception for errors raised while parsing OPENQASM.
"""

from qiskit.exceptions import QiskitError


class QasmError(QiskitError):
    """Base class for errors raised while parsing OPENQASM."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
