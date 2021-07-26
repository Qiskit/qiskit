# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions for ClassicalFunction compiler"""

from qiskit.exceptions import QiskitError


class ClassicalFunctionCompilerError(QiskitError):
    """ClassicalFunction compiler generic error."""

    pass


class ClassicalFunctionParseError(ClassicalFunctionCompilerError):
    """ClassicalFunction compiler parse error.
    The classicalfunction function fails at parsing time."""

    pass


class ClassicalFunctionCompilerTypeError(ClassicalFunctionCompilerError):
    """ClassicalFunction compiler type error.
    The classicalfunction function fails at type checking time."""

    pass
