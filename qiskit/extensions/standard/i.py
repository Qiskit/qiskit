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

"""
Identity gate.
"""
import warnings
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.util import deprecate_arguments


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

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{I}} =
            \begin{bmatrix}
                1 & 0 \\
                0 & 1
            \end{bmatrix}
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


@deprecate_arguments({'q': 'qubit'})
def i(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply Identity to to a specified qubit (qubit).

    The Identity gate ensures that nothing is applied to a qubit for one unit
    of gate time. It leaves the quantum states |0> and |1> unchanged.
    The Identity gate should not be optimized or unrolled (it is an opaque gate).

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.id(0)  # or circuit.i(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.i import IGate
            IGate().to_matrix()
    """
    return self.append(IGate(), [qubit], [])


@deprecate_arguments({'q': 'qubit'})
def iden(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Deprecated identity gate."""
    warnings.warn('The QuantumCircuit.iden() method is deprecated as of 0.14.0, and '
                  'will be removed no earlier than 3 months after that release date. '
                  'You should use the QuantumCircuit.i() method instead.',
                  DeprecationWarning, stacklevel=2)
    return self.append(IGate(), [qubit], [])


# support both i and id as methods of QuantumCircuit
QuantumCircuit.i = i
QuantumCircuit.id = i

QuantumCircuit.iden = iden  # deprecated, remove once IdGate is removed
