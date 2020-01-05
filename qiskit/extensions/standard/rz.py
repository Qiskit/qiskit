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
Rotation around the z-axis.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate


class RZGate(Gate):
    """rotation around the z-axis."""

    def __init__(self, phi):
        """Create new rz single qubit gate."""
        super().__init__("rz", 1, [phi])

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(self.params[0]), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate.

        rz(phi)^dagger = rz(-phi)
        """
        return RZGate(-self.params[0])


def rz(self, phi, q):  # pylint: disable=invalid-name
    """Apply Rz gate with angle phi to a specified qubit (q).
    The Rz gate corresponds to a rotation of phi radians from |+> about
    the z-axis on the Bloch sphere.

    Example:
    circuit = QuantumCircuit(1)
    phi = numpy.pi/2
    circuit.ry(numpy.pi/2,0) # This brings the quantum state from |0> to |+>
    circuit.rz(phi,0)
    circuit.draw()
            ┌──────────┐┌──────────┐
    q_0: |0>┤ Ry(pi/2) ├┤ Rz(pi/2) ├
            └──────────┘└──────────┘
    Resulting Statevector:
    [ 0.707+0j, 0+0.707j ]
    """
    return self.append(RZGate(phi), [q], [])


QuantumCircuit.rz = rz
