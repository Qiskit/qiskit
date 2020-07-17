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

"""Identity gate."""

import warnings
import numpy
from qiskit.circuit.gate import Gate


class IMeta(type):
    """A metaclass to ensure that Id and I are of the same type.

    Can be removed when IdGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {IGate, IdGate}  # pylint: disable=unidiomatic-typecheck


class IGate(Gate, metaclass=IMeta):
    r"""Identity gate.

    Identity gate corresponds to a single-qubit gate wait cycle,
    and should not be optimized or unrolled (it is an opaque gate).

    **Matrix Representation:**

    .. math::

        I = \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::
             ┌───┐
        q_0: ┤ I ├
             └───┘
    """

    def __init__(self, label=None):
        """Create new Identity gate."""
        super().__init__('id', 1, [], label=label)

    def inverse(self):
        """Invert this gate."""
        return IGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the identity gate."""
        return numpy.array([[1, 0],
                            [0, 1]], dtype=complex)


class IdGate(IGate, metaclass=IMeta):
    """The deprecated IGate class."""

    def __init__(self):
        warnings.warn('The class IdGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class IGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()
