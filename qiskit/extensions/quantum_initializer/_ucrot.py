# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
(Abstract) base class for uniformly controlled (also called multiplexed) single-qubit rotations R_t.
This class provides a basis for the decomposition of uniformly controlled R_y and R_z gates (i.e., for t=y,z).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_t(a_i) is applied to the target qubit for a (real) angle a_i.
"""

import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.rz import RZGate
from qiskit.extensions.standard.ry import RYGate

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCRot(CompositeGate):
    """
    Uniformly controlled rotations (also called multiplexed rotations). The decomposition is based on
    'Synthesis of Quantum Logic Circuits' by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)

    Input:
    angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]

   q_controls = list of k control qubits (or empty list if no controls). The control qubits are ordered according to
                their significance in increasing order:
                For example if q_controls=[q[1],q[2]] (with q = QuantumRegister(2)), the rotation R_t(a_0)is
                performed if q[1] and q[2] are in the state zero, the rotation  R_t(a_1) is performed if
                q[1] is in the state one and q[2] is in the state zero, and so on.

    q_target =  target qubit, where we act on with the single-qubit gates.

    circ =      QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, angle_list, q_controls, q_target, rot_axis, circ=None):
        self.q_controls = q_controls
        self.q_target = q_target
        self.rot_axes = rot_axis

        """Check types"""
        # Check if q_controls has type "list"
        if not type(q_controls) == list:
            raise QiskitError(
                "The control qubits must be provided as a list (also if there is only one control qubit).")
        # Check if angle_list has type "list"
        if not type(angle_list) == list:
            raise QiskitError(
                "The angles are not provided in a list.")
        # Check if the angles in angle_list are real numbers
        for a in angle_list:
            try:
                float(a)
            except:
                raise QiskitError("An angle cannot be converted to type float.")
        # Check if there is one target qubit provided
        if not (type(q_target) == tuple and type(q_target[0]) == QuantumRegister):
            raise QiskitError("The target qubit is not a single qubit from a QuantumRegister.")

        """Check input form"""
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError("The number of controlled rotation gates is not a non-negative power of 2.")
        # Check if number of control qubits does correspond to the number of rotations
        if num_contr != len(q_controls):
            raise QiskitError("Number of controlled rotations does not correspond to the number of control-qubits.")
        if rot_axis != "Y" and rot_axis != "Z":
            raise QiskitError("Rotation axis is not supported.")
        # Create new composite gate.
        num_qubits = len(q_controls) + len(q_target)
        self.num_qubits = int(num_qubits)
        qubits = q_controls + [q_target]
        super().__init__("init", angle_list, qubits, circ)
        # Check that the target qubis is not also listed as an control qubit
        self._check_dups(qubits, message="The target qubit cannot also be listed as a control qubit.")

    # finds a decomposition of a UC rotation gate into elementary gates (C-NOTs and single-qubit rotations).
    def _dec_ucrot(self):
        """
        Call to populate the self.data list with gates that implement the uniformly controlled rotation
        """
        if len(self.q_controls) == 0:
            if self.rot_axes == "Z":
                if np.abs(self.params[0]) > _EPS:
                    self._attach(RZGate(self.params[0], self.q_target))
            if self.rot_axes == "Y":
                if np.abs(self.params[0]) > _EPS:
                    self._attach(RYGate(self.params[0], self.q_target))
        else:
            # First, we find the rotation angles of the single-qubit rotations acting on the target qubit
            angles = self.params.copy()
            self._dec_uc_rotations(angles, 0, len(angles), False)
            # Now, it is easy to place the C-NOT gates to get back the full decomposition.
            for i in range(len(angles)):
                if self.rot_axes == "Z":
                    if np.abs(angles[i]) > _EPS:
                        self._attach(RZGate(angles[i], self.q_target))
                if self.rot_axes == "Y":
                    if np.abs(angles[i]) > _EPS:
                        self._attach(RYGate(angles[i], self.q_target))
                # Determine the index of the qubit we want to control the C-NOT gate. Note that it corresponds
                # to the number of trailing zeros in the binary representaiton of i+1
                if not i == len(angles) - 1:
                    binary_rep = np.binary_repr(i + 1)
                    q_contr_index = len(binary_rep) - len(binary_rep.rstrip('0'))
                else:
                    # Handle special case:
                    q_contr_index = len(self.q_controls) - 1
                self._attach(CnotGate(self.q_controls[q_contr_index], self.q_target))

    # Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at the end of the circuit.
    # The rotation angles of the gate R_t are stored in angles[start_index:end_index].
    # If reversed == True, it decomposes the gate such that there is a C-NOT gate at the start of the circuit
    # (in fact, the circuit topology for the reversed decomposition is the reversed one of the original decomposition)
    def _dec_uc_rotations(self, angles, start_index, end_index, reversedDec):
        interval_len_half = (end_index - start_index)//2
        for i in range(start_index, start_index + interval_len_half):
            if not reversedDec:
                angles[i], angles[i + interval_len_half] = self._update_angles(angles[i], angles[i + interval_len_half])
            else:
                angles[i + interval_len_half], angles[i] = self._update_angles(angles[i], angles[i + interval_len_half])
        if interval_len_half <= 1:
            return
        else:
            self._dec_uc_rotations(angles, start_index, start_index + interval_len_half, False)
            self._dec_uc_rotations(angles, start_index + interval_len_half, end_index, True)

    # Calculate the new rotation angles according to Shende's decomposition
    def _update_angles(self, a1, a2):
        return (a1 + a2) / 2.0, (a1 - a2) / 2.0
