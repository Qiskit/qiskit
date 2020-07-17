# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Global Mølmer–Sørensen gate.
"""

from typing import Union, List

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXXGate


class GMS(QuantumCircuit):
    r"""Global Mølmer–Sørensen gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1   GMS    ├
             │           │
        q_2: ┤2          ├
             └───────────┘

    **Expanded Circuit:**

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import GMS
        import qiskit.tools.jupyter
        import numpy as np
        circuit = GMS(num_qubits=3, theta=[[0, np.pi/4, np.pi/8],
                                           [0, 0, np.pi/2],
                                           [0, 0, 0]])
        %circuit_library_info circuit.decompose()

    The Mølmer–Sørensen gate is native to ion-trap systems. The global MS
    can be applied to multiple ions to entangle multiple qubits simultaneously [1].

    In the two-qubit case, this is equivalent to an XX(theta) interaction,
    and is thus reduced to the RXXGate. The global MS gate is a sum of XX
    interactions on all pairs [2].

    .. math::

        GMS(\chi_{12}, \chi_{13}, ..., \chi_{n-1 n}) =
        exp(-i \sum_{i=1}^{n} \sum_{j=i+1}^{n} X{\otimes}X \frac{\chi_{ij}}{2})

    **References:**

    [1] Sørensen, A. and Mølmer, K., Multi-particle entanglement of hot trapped ions.
    Physical Review Letters. 82 (9): 1835–1838.
    `arXiv:9810040 <https://arxiv.org/abs/quant-ph/9810040>`_

    [2] Maslov, D. and Nam, Y., Use of global interactions in efficient quantum circuit
    constructions. New Journal of Physics, 20(3), p.033018.
    `arXiv:1707.06356 <https://arxiv.org/abs/1707.06356>`_
    """

    def __init__(self,
                 num_qubits: int,
                 theta: Union[List[List[float]], np.ndarray]) -> None:
        """Create a new Global Mølmer–Sørensen (GMS) gate.

        Args:
            num_qubits: width of gate.
            theta: a num_qubits x num_qubits symmetric matrix of
                interaction angles for each qubit pair. The upper
                triangle is considered.
        """
        super().__init__(num_qubits, name="gms")
        if not isinstance(theta, list):
            theta = [theta] * int((num_qubits**2 - 1) / 2)
        gms = QuantumCircuit(num_qubits, name="gms")
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                gms.append(RXXGate(theta[i][j]), [i, j])
        self.append(gms, self.qubits)
