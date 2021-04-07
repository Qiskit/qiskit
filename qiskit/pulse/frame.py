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

"""Implements a Frame."""

from typing import Union

from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.utils import validate_index


class Frame:
    """A frame is a frequency and a phase."""
    prefix = 'f'

    def __init__(self, index: Union[int, Parameter]):
        """
        Args:
            index: The index of the frame.
        """
        validate_index(index)
        self._index = index
        self._hash = hash((type(self), self._index))

    @property
    def index(self) -> Union[int, ParameterExpression]:
        """Return the index of this frame. The index is a label for a frame."""
        return self._index

    @property
    def name(self) -> str:
        """Return the shorthand alias for this frame, which is based on its type and index."""
        return '{}{}'.format(self.__class__.prefix, self._index)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._index})'

    def __eq__(self, other: 'Frame') -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same index.

        Args:
            other: The frame to compare to this frame.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self.index == other.index

    def __hash__(self):
        return self._hash
