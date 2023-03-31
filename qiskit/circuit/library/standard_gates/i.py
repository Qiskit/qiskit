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

from typing import Optional
import numpy
from qiskit.circuit.gate import Gate


class IGate(Gate):
    r"""Identity gate.

    This gate is a quantum no-op and represents an identity channel on a single qubit.  It will not
    have a logical effect on the qubit state, and will likely be optimized away by the compiler at
    all optimization levels (except zero, if ``id`` is specified in the basis of the target).
    Certain hardware backends *may* choose to implement this instruction with a potentially non-zero
    idle duration, for example a delay of length equal to that of a single-qubit gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit` with the
    :meth:`~qiskit.circuit.QuantumCircuit.i` and :meth:`~qiskit.circuit.QuantumCircuit.id` methods.

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

    def __init__(self, label: Optional[str] = None):
        """Create new Identity gate."""
        super().__init__("id", 1, [], label=label)

    def _define(self) -> None:
        from qiskit.circuit import QuantumCircuit  # pylint: disable=cyclic-import

        self.definition = QuantumCircuit(1, name=self.name)

    def inverse(self):
        """Invert this gate."""
        return IGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the identity gate."""
        return numpy.array([[1, 0], [0, 1]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        return IGate()
