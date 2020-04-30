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

"""
Implementation of the abstract class UCPauliRotGate for uniformly controlled
(also called multiplexed) single-qubit rotations
around the Y-axes (i.e., uniformly controlled R_y rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_y(a_i) is applied to the target qubit.
"""
import math

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer.uc_pauli_rot import UCPauliRotGate, UCPauliRotMeta


class UCRYMeta(UCPauliRotMeta):
    """A metaclass to ensure that UCRYGate and UCY are of the same type.

    Can be removed when UCY gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {UCRYGate, UCY}  # pylint: disable=unidiomatic-typecheck


class UCRYGate(UCPauliRotGate, metaclass=UCRYMeta):
    """
    Uniformly controlled rotations (also called multiplexed rotations).
    The decomposition is based on
    'Synthesis of Quantum Logic Circuits' by V. Shende et al.
    (https://arxiv.org/pdf/quant-ph/0406176.pdf)

    Input:
    angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]
    """

    def __init__(self, angle_list):
        super().__init__(angle_list, "Y")


def ucry(self, angle_list, q_controls, q_target):
    """Attach a uniformly controlled (also called multiplexed) Ry rotation gate to a circuit.

    The decomposition is base on https://arxiv.org/pdf/quant-ph/0406176.pdf by Shende et al.

    Args:
        angle_list (list[numbers): list of (real) rotation angles :math:`[a_0,...,a_{2^k-1}]`
        q_controls (QuantumRegister|list[Qubit]): list of k control qubits
            (or empty list if no controls). The control qubits are ordered according to their
            significance in increasing order: For example if ``q_controls=[q[0],q[1]]``
            (with ``q = QuantumRegister(2)``), the rotation ``Ry(a_0)`` is performed if ``q[0]``
            and ``q[1]`` are in the state zero, the rotation ``Ry(a_1)`` is performed if ``q[0]``
            is in the state one and ``q[1]`` is in the state zero, and so on
        q_target (QuantumRegister|Qubit): target qubit, where we act on with
            the single-qubit rotation gates

    Returns:
        QuantumCircuit: the uniformly controlled rotation gate is attached to the circuit.

    Raises:
        QiskitError: if the list number of control qubits does not correspond to the provided
            number of single-qubit unitaries; if an input is of the wrong type
    """

    if isinstance(q_controls, QuantumRegister):
        q_controls = q_controls[:]
    if isinstance(q_target, QuantumRegister):
        q_target = q_target[:]
        if len(q_target) == 1:
            q_target = q_target[0]
        else:
            raise QiskitError("The target qubit is a QuantumRegister containing"
                              " more than one qubits.")
    # Check if q_controls has type "list"
    if not isinstance(angle_list, list):
        raise QiskitError("The angles must be provided as a list.")
    num_contr = math.log2(len(angle_list))
    if num_contr < 0 or not num_contr.is_integer():
        raise QiskitError("The number of controlled rotation gates is not"
                          " a non-negative power of 2.")
    # Check if number of control qubits does correspond to the number of rotations
    if num_contr != len(q_controls):
        raise QiskitError("Number of controlled rotations does not correspond to"
                          " the number of control-qubits.")
    return self.append(UCRYGate(angle_list), [q_target] + q_controls, [])


class UCY(UCRYGate, metaclass=UCRYMeta):
    """The deprecated UCRYGate class."""

    def __init__(self, angle_list):
        import warnings
        warnings.warn('The class UCY is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class UCRYGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(angle_list)


def ucy(self, angle_list, q_controls, q_target):
    """Deprecated version of ucry."""
    import warnings
    warnings.warn('The QuantumCircuit ucy() method is deprecated as of 0.14.0, and '
                  'will be removed no earlier than 3 months after that release date. '
                  'You should use the QuantumCircuit ucry() method instead.',
                  DeprecationWarning, stacklevel=2)
    return ucry(self, angle_list, q_controls, q_target)


QuantumCircuit.ucry = ucry
QuantumCircuit.ucy = ucy  # deprecated, but still supported
