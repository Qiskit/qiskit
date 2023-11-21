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

"""Exception for errors raised while handling OpenQASM 2.0."""

# Re-export from the new place to ensure that old code continues to work.
from qiskit.qasm2.exceptions import QASM2Error as QasmError  # pylint: disable=unused-import
