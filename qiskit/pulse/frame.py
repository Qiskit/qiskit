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

from typing import Any, Set, Union
import numpy as np

from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError


class Frame:
    """A frame is a frequency and a phase."""
    prefix = 'f'

    def __init__(self, index: Union[int, Parameter]):
        """
        Args:
            index: The index of the frame.
        """
        self._validate_index(index)
        self._index = index
        self._hash = None

        self._parameters = set()
        if isinstance(index, ParameterExpression):
            self._parameters.update(index.parameters)

    @property
    def index(self) -> Union[int, ParameterExpression]:
        """Return the index of this frame. The index is a label for a frame."""
        return self._index

    @staticmethod
    def _validate_index(index: Any) -> None:
        """
        Raise a PulseError if the Frame index is invalid, namely, if it's not a positive
        integer.

        Raises:
            PulseError: If ``index`` is not a non-negative integer.
        """
        if isinstance(index, ParameterExpression) and index.parameters:
            # Parameters are unbound
            return
        elif isinstance(index, ParameterExpression):
            index = float(index)
            if index.is_integer():
                index = int(index)

        if not isinstance(index, (int, np.integer)) and index < 0:
            raise PulseError('Frame index must be a nonnegative integer')

    @property
    def name(self) -> str:
        """Return the shorthand alias for this frame, which is based on its type and index."""
        return '{}{}'.format(self.__class__.prefix, self._index)

    @property
    def parameters(self) -> Set:
        """Parameters which determine the frame index."""
        return self._parameters

    def is_parameterized(self) -> bool:
        """Return True iff the frame is parameterized."""
        return bool(self.parameters)

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
        if self._hash is None:
            self._hash = hash((type(self), self._index))
        return self._hash
