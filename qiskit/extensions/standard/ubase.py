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
Element of SU(2).
"""
import warnings
import numpy

from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit


class UBase(Gate):
    """Element of SU(2)."""

    def __init__(self, theta, phi, lam):
        warnings.warn('UBase is deprecated and it will be removed after 0.9. '
                      'Use U3Gate instead.', DeprecationWarning, 2)
        super().__init__("U", 1, [theta, phi, lam])

    def inverse(self):
        """Invert this gate.

        U(theta,phi,lambda)^dagger = U(-theta,-lambda,-phi)
        """
        warnings.warn('UBase.inverse is deprecated and it will be removed after 0.9. '
                      'Use U3Gate.inverse instead.',
                      DeprecationWarning, 2)
        return UBase(-self.params[0], -self.params[2], -self.params[1])

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
        warnings.warn('UBase.to_matrix is deprecated and it will be removed after 0.9.'
                      'Use U3Gate.to_matrix instead.',
                      DeprecationWarning, 2)

        theta, phi, lam = self.params
        return numpy.array(
            [[
                numpy.cos(theta / 2),
                -numpy.exp(1j * lam) * numpy.sin(theta / 2)
            ],
             [
                 numpy.exp(1j * phi) * numpy.sin(theta / 2),
                 numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2)
             ]],
            dtype=complex)


def u_base(self, theta, phi, lam, q):
    """Apply U to q."""
    return self.append(UBase(theta, phi, lam), [q], [])


QuantumCircuit.u_base = u_base
