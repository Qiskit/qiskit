# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A directive instruction."""

from .instruction import Instruction


class Directive(Instruction):
    """Directive quantum instruction.

    This is a base class that is intended to be used for instructions which
    should be treated as directives.

    All directives are treated like Barrier instructions by the transpiler
    and drawer.
    """

    def __init__(self, name, num_qubits, num_clbits, params):
        """Create a new instruction.

        Args:
            name (str): instruction name
            num_qubits (int): instruction's qubit width
            num_clbits (int): instruction's clbit width
            params (list[int|float|complex|str|ndarray|list|ParameterExpression]):
                list of parameters
        """
        # pylint: disable = useless-super-delegation
        # Duration and dt are used as default values in parent class
        super().__init__(name, num_qubits, num_clbits, params)
