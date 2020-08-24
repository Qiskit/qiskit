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


"""
Global R gates.
"""

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RGate, RZGate


class GR(QuantumCircuit):
    r"""Global R gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌──────────┐
        q_0: ┤0         ├
             │          │
        q_1: ┤1 GR(ϴ,φ) ├
             │          │
        q_2: ┤2         ├
             └──────────┘

    **Expanded Circuit:**

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import GR
        import qiskit.tools.jupyter
        import numpy as np
        circuit = GR(num_qubits=3, theta=np.pi/4, phi=np.pi/2)
        %circuit_library_info circuit.decompose()

    The global R gate is native to atomic systems (ion traps, cold neutrals). The global R
    can be applied to multiple qubits simultaneously.

    In the one-qubit case, this is equivalent to an R(theta, phi) interaction,
    and is thus reduced to the RGate. The global R gate is a sum of Rs
    interactions on all individual qubits.

    .. math::

        GR(\theta, \phi) =
        exp(-i \sum_{i=1}^{n} (\cos(\phi)X + sin(\phi)Y) \theta/2)

    """
    def __init__(self,
                 num_qubits: int,
                 theta: float,
                 phi: float) -> None:
        """Create a new Global R (GR) gate.

        Args:
            num_qubits: number of qubits.
            theta: rotation angle about axis determined by phi
            phi: angle of rotation axis in xy-plane
        """
        super().__init__(num_qubits, name="gr")
        gr = QuantumCircuit(num_qubits, name="gr")
        for i in range(self.num_qubits):
            gr.append(RGate(theta, phi), [i])
        self.append(gr, self.qubits)

class GRx(GR):
    def __init__(self, 
                 num_qubits: int,
                 theta: float) -> None:
        """Create a new Global Rx (GRx) gate.

        Args:
            num_qubits: number of qubits.
            theta: rotation angle about x-axis
        """
        super().__init__(num_qubits, theta, phi=0, name='grx')

class GRy(GR):
    def __init__(self, 
                 num_qubits: int,
                 theta: float) -> None:
        """Create a new Global Ry (GRy) gate.

        Args:
            num_qubits: number of qubits.
            theta: rotation angle about y-axis
        """
        super().__init__(num_qubits, theta, phi=np.pi/2, name='gry',)

class GRz(QuantumCircuit):
    def __init__(self,
                 num_qubits: int,
                 phi: float) -> None:
        """Create a new Global Rz (GRz) gate.

        Args:
            num_qubits: number of qubits.
            phi: rotation angle about z-axis 
        """
        super().__init__(num_qubits, name="grz")
        grz = QuantumCircuit(num_qubits, name="grz")
        for i in range(self.num_qubits):
            gr.append(RzGate(phi), [i])
        self.append(grz, self.qubits)

