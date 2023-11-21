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

"""
===============================================
Top-level exceptions (:mod:`qiskit.exceptions`)
===============================================

All Qiskit-related errors raised by Qiskit are subclasses of the base:

.. autoexception:: QiskitError

.. note::

    Errors that are just general programming errors, such as incorrect typing, may still raise
    standard Python errors such as ``TypeError``.  :exc:`QiskitError` is generally for errors raised
    in usage that is particular to Qiskit.

Many of the Qiskit subpackages define their own more granular error, to help in catching only the
subset of errors you care about.  For example, :mod:`qiskit.circuit` almost exclusively uses
:exc:`.CircuitError`, while both :exc:`.QASM2ExportError` and :exc:`.QASM2ParseError` derive from
:exc:`.QASM2Error` in :mod:`qiskit.qasm2`, which is in turn a type of :exc:`.QiskitError`.

Qiskit has several optional features that depend on other packages that are not required for a
minimal install.  You can read more about those, and ways to check for their presence, in
:mod:`qiskit.utils.optionals`.  Trying to use a feature that requires an optional extra will raise a
particular error, which subclasses both :exc:`QiskitError` and the Python built-in ``ImportError``.

.. autoexception:: MissingOptionalLibraryError

Two more uncommon errors relate to failures in reading user-configuration files, or specifying a
filename that cannot be used:

.. autoexception:: QiskitUserConfigError
.. autoexception:: InvalidFileError
"""

from typing import Optional


class QiskitError(Exception):
    """Base class for errors raised by Qiskit."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class QiskitUserConfigError(QiskitError):
    """Raised when an error is encountered reading a user config file."""

    message = "User config invalid"


class MissingOptionalLibraryError(QiskitError, ImportError):
    """Raised when an optional library is missing."""

    def __init__(
        self, libname: str, name: str, pip_install: Optional[str] = None, msg: Optional[str] = None
    ) -> None:
        """Set the error message.
        Args:
            libname: Name of missing library
            name: Name of class, function, module that uses this library
            pip_install: pip install command, if any
            msg: Descriptive message, if any
        """
        message = [f"The '{libname}' library is required to use '{name}'."]
        if pip_install:
            message.append(f"You can install it with '{pip_install}'.")
        if msg:
            message.append(f" {msg}.")

        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self) -> str:
        """Return the message."""
        return repr(self.message)


class InvalidFileError(QiskitError):
    """Raised when the file provided is not valid for the specific task."""
