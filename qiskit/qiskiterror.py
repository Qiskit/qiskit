# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by the Qiskit.
"""


# NOTE(mtreinish): This class is here to maintain backwards compatibility and should not be
# used directly. Instead use the QiskitError class.
class QISKitError(Exception):
    """Old Base class for errors raised by the Qiskit for backwards compat only, not for use."""


class QiskitError(QISKitError):
    """Base class for errors raised by the Qiskit."""

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
