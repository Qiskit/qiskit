# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions and warnings relating to QASM 3."""

__all__ = ("QASM3Warning",)


class QASM3Warning(UserWarning):
    """A warning raised when something questionable happens while interacting with the OpenQASM 3
    converters."""
