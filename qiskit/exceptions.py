# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions for errors raised by Qiskit."""

from typing import Optional


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


class MissingOptionalLibraryError(QiskitError):
    """Raised when an optional library is missing."""

    def __init__(self,
                 libname: str,
                 name: str,
                 pip_install: Optional[str] = None,
                 msg: Optional[str] = None) -> None:
        """Set the error message.
        Args:
            libname: Name of missing library
            name: Name of class, function, module that uses this library
            pip_install: pip install command, if any
            msg: Descriptive message, if any
        """
        message = ["The '{}' library is required to use '{}'.".format(libname, name)]
        if pip_install:
            message.append("You can install it with '{}'.".format(pip_install))
        if msg:
            message.append(' {}.'.format(msg))

        super().__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self) -> str:
        """Return the message."""
        return repr(self.message)
