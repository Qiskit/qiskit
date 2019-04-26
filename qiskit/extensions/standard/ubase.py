# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Element of SU(2).
"""
import numpy
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.decorators import _op_expand, _to_bits


class UBase(Gate):  # pylint: disable=abstract-method
    """Element of SU(2)."""

    def __init__(self, theta, phi, lam):
        super().__init__("U", 1, [theta, phi, lam])

    def inverse(self):
        """Invert this gate.

        U(theta,phi,lambda)^dagger = U(-theta,-lambda,-phi)
        """
        return UBase(-self.params[0], -self.params[2], -self.params[1])

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
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


@_to_bits(1)
@_op_expand(1)
def u_base(self, theta, phi, lam, q):
    """Apply U to q."""
    return self.append(UBase(theta, phi, lam), [q], [])


QuantumCircuit.u_base = u_base
CompositeGate.u_base = u_base
