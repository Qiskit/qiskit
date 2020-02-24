# -*- coding: utf-8 -*-

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

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
(Abstract) base class for uniformly controlled (also called multiplexed) single-qubit rotations R_t.
This class provides a basis for the decomposition of uniformly controlled R_x,R_y and R_z gates
(i.e., for t=x,y,z). These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_t(a_i) is applied to the target qubit for a (real) angle a_i.
"""

import math

import numpy as np

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.quantumcircuit import QuantumRegister
from qiskit.exceptions import QiskitError

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCPauliRotMeta(type):
    """A metaclass to ensure that UCPauliRotGate and UCRot are of the same type.

    Can be removed when UCRot gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {UCPauliRotGate, UCRot}  # pylint: disable=unidiomatic-typecheck


class UCPauliRotGate(Gate, metaclass=UCPauliRotMeta):
    """
    Uniformly controlled rotations (also called multiplexed rotations).
    The decomposition is based on 'Synthesis of Quantum Logic Circuits'
    by Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)

    Input:
    angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]. Must have at least one entry.

    rot_axis = rotation axis for the single qubit rotations
               (currently, 'X', 'Y' and 'Z' are supported)
    """

    def __init__(self, angle_list, rot_axis):
        self.rot_axes = rot_axis
        # Check if angle_list has type "list"
        if not isinstance(angle_list, list):
            raise QiskitError('The angles are not provided in a list.')
        # Check if the angles in angle_list are real numbers
        for angle in angle_list:
            try:
                float(angle)
            except TypeError:
                raise QiskitError(
                    'An angle cannot be converted to type float (real angles are expected).')
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError(
                'The number of controlled rotation gates is not a non-negative power of 2.')
        if rot_axis not in ('X', 'Y', 'Z'):
            raise QiskitError('Rotation axis is not supported.')
        # Create new gate.
        num_qubits = int(num_contr) + 1
        super().__init__('ucr' + rot_axis.lower(), num_qubits, angle_list)

    def _define(self):
        ucr_circuit = self._dec_ucrot()
        gate = ucr_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        ucr_circuit = QuantumCircuit(q)
        ucr_circuit.append(gate, q[:])
        self.definition = ucr_circuit.data

    def _dec_ucrot(self):
        """
        Finds a decomposition of a UC rotation gate into elementary gates
        (C-NOTs and single-qubit rotations).
        """
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        q_target = q[0]
        q_controls = q[1:]
        if not q_controls:  # equivalent to: if len(q_controls) == 0
            if self.rot_axes == 'X':
                if np.abs(self.params[0]) > _EPS:
                    circuit.rx(self.params[0], q_target)
            if self.rot_axes == 'Y':
                if np.abs(self.params[0]) > _EPS:
                    circuit.ry(self.params[0], q_target)
            if self.rot_axes == 'Z':
                if np.abs(self.params[0]) > _EPS:
                    circuit.rz(self.params[0], q_target)
        else:
            # First, we find the rotation angles of the single-qubit rotations acting
            #  on the target qubit
            angles = self.params.copy()
            UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
            # Now, it is easy to place the C-NOT gates to get back the full decomposition.
            for (i, angle) in enumerate(angles):
                if self.rot_axes == 'X':
                    if np.abs(angle) > _EPS:
                        circuit.rx(angle, q_target)
                if self.rot_axes == 'Y':
                    if np.abs(angle) > _EPS:
                        circuit.ry(angle, q_target)
                if self.rot_axes == 'Z':
                    if np.abs(angle) > _EPS:
                        circuit.rz(angle, q_target)
                # Determine the index of the qubit we want to control the C-NOT gate.
                # Note that it corresponds
                # to the number of trailing zeros in the binary representation of i+1
                if not i == len(angles) - 1:
                    binary_rep = np.binary_repr(i + 1)
                    q_contr_index = len(binary_rep) - \
                        len(binary_rep.rstrip('0'))
                else:
                    # Handle special case:
                    q_contr_index = len(q_controls) - 1
                # For X rotations, we have to additionally place some Ry gates around the
                # C-NOT gates. They change the basis of the NOT operation, such that the
                # decomposition of for uniformly controlled X rotations works correctly by symmetry
                # with the decomposition of uniformly controlled Z or Y rotations
                if self.rot_axes == 'X':
                    circuit.ry(np.pi / 2, q_target)
                circuit.cx(q_controls[q_contr_index], q_target)
                if self.rot_axes == 'X':
                    circuit.ry(-np.pi / 2, q_target)
        return circuit

    @staticmethod
    def _dec_uc_rotations(angles, start_index, end_index, reversed_dec):
        """
        Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at
        the end of the circuit. The rotation angles of the gate R_t are stored in
        angles[start_index:end_index]. If reversed_dec == True, it decomposes the gate such that
        there is a C-NOT gate at the start of the circuit (in fact, the circuit topology for
        the reversed decomposition is the reversed one of the original decomposition)
        """
        interval_len_half = (end_index - start_index) // 2
        for i in range(start_index, start_index + interval_len_half):
            if not reversed_dec:
                angles[i], angles[i + interval_len_half] = \
                    UCPauliRotGate._update_angles(
                        angles[i], angles[i + interval_len_half])
            else:
                angles[i + interval_len_half], angles[i] = \
                    UCPauliRotGate._update_angles(
                        angles[i], angles[i + interval_len_half])
        if interval_len_half <= 1:
            return
        else:
            UCPauliRotGate._dec_uc_rotations(angles, start_index, start_index + interval_len_half,
                                             False)
            UCPauliRotGate._dec_uc_rotations(angles, start_index + interval_len_half, end_index,
                                             True)

    @staticmethod
    def _update_angles(angle1, angle2):
        """Calculate the new rotation angles according to Shende's decomposition."""
        return (angle1 + angle2) / 2.0, (angle1 - angle2) / 2.0


class UCRot(UCPauliRotGate, metaclass=UCPauliRotMeta):
    """The deprecated DiagonalGate class."""

    def __init__(self, angle_list, rot_axis):
        import warnings
        warnings.warn('The class UCRot is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class UCPauliRotGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(angle_list, rot_axis)
