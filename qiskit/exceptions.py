# -*- coding: utf-8 -*-

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

"""Exceptions for errors raised by Qiskit."""


class QiskitError(Exception):
    """Base class for errors raised by Qiskit."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class QiskitIndexError(QiskitError, IndexError):
    """Raised when a sequence subscript is out of range."""
    pass


class QiskitUserConfigError(QiskitError):
    """Raised when an error is encountered reading a user config file."""
    message = "User config invalid"
