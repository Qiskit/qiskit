# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Implementation of the abstract class UCRot for uniformly controlled (also called multiplexed) single-qubit rotations
around the Y-axes (i.e., uniformly controlled R_y rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_y(a_i) is applied to the target qubit.
"""

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.extensions.quantum_initializer._ucrot import UCRot


class UCY(UCRot):  # pylint: disable=abstract-method
    """
       Uniformly controlled rotations (also called multiplexed rotations). The decomposition is based on
       'Synthesis of Quantum Logic Circuits' by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)

       Input:
       angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]

      q_controls = list of k control qubits (or empty list if no controls). The control qubits are ordered according to
                   their significance in increasing order:
                   For example if q_controls=[q[1],q[2]] (with q = QuantumRegister(2)), the rotation R_y(a_0)is
                   performed if q[1] and q[2] are in the state zero, the rotation  R_y(a_1) is performed if
                   q[1] is in the state one and q[2] is in the state zero, and so on.

       q_target =  target qubit, where we act on with the single-qubit gates.

       circ =      QuantumCircuit or CompositeGate containing this gate
       """

    def __init__(self, angle_list, q_controls, q_target, circ=None):
        super().__init__(angle_list, q_controls, q_target, "Y", circ)
        # call to generate the circuit that takes the desired vector to zero
        self._dec_ucrot()


def ucy(self, angle_list, q_controls, q_target):
    return self._attach(UCY(angle_list, q_controls, q_target))


QuantumCircuit.ucy = ucy
CompositeGate.ucy = ucy
