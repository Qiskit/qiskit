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
Clifford template 3_1:
.. parsed-literal::

             ┌───┐┌───┐┌───┐
        q_0: ┤ S ├┤ S ├┤ Z ├
             └───┘└───┘└───┘
"""

from qiskit.circuit.quantumcircuit import QuantumCircuit


def clifford_3_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.s(0)
    qc.z(0)
    return qc
