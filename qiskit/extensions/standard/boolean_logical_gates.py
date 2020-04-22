# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Boolean Logical AND and OR Gates."""

import warnings
from qiskit.circuit import QuantumCircuit


def logical_and(self, qr_variables, qb_target, qr_ancillae, flags=None, mct_mode='no-ancilla'):
    """Build a collective conjunction (AND) circuit in place using mct.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to build the conjunction on.
        qr_variables (QuantumRegister): The QuantumRegister holding the variable qubits.
        qb_target (Qubit): The target qubit to hold the conjunction result.
        qr_ancillae (QuantumRegister): The ancillary QuantumRegister for building the mct.
        flags (list[int]): A list of +1/-1/0 to mark negations or omissions of qubits.
        mct_mode (str): The mct building mode.
    """
    # pylint: disable=cyclic-import
    from qiskit.circuit.library import AND

    warnings.warn('The QuantumCircuit.AND method is deprecated as of Terra 0.13.1 / Aqua 0.7.0 and '
                  'will be removed no earlier than 3 months after the release date. '
                  'The logic AND has moved to qiskit.circuit.library.AND and has become a circuit '
                  'object which can be appended to your existing circuit.',
                  DeprecationWarning, stacklevel=2)
    and_circuit = AND(num_variable_qubits=len(qr_variables), flags=flags, mcx_mode=mct_mode)
    qubits = qr_variables[:] + [qb_target]
    if qr_ancillae:
        qubits += qr_ancillae[:and_circuit.num_ancilla_qubits]

    self.append(and_circuit.to_gate(), qubits)


def logical_or(self, qr_variables, qb_target, qr_ancillae, flags=None, mct_mode='basic'):
    """Build a collective disjunction (OR) circuit in place using mct.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to build the disjunction on.
        qr_variables (QuantumRegister): The QuantumRegister holding the variable qubits.
        flags (list[int]): A list of +1/-1/0 to mark negations or omissions of qubits.
        qb_target (Qubit): The target qubit to hold the disjunction result.
        qr_ancillae (QuantumRegister): The ancillary QuantumRegister for building the mct.
        mct_mode (str): The mct building mode.
    """
    # pylint: disable=cyclic-import
    from qiskit.circuit.library import OR

    warnings.warn('The QuantumCircuit.OR method is deprecated as of Terra 0.13.1 / Aqua 0.7.0 and '
                  'will be removed no earlier than 3 months after the release date. '
                  'The logic OR has moved to qiskit.circuit.library.OR and has become a circuit '
                  'object which can be appended to your existing circuit.',
                  DeprecationWarning, stacklevel=2)
    or_circuit = OR(num_variable_qubits=len(qr_variables), flags=flags, mcx_mode=mct_mode)
    qubits = qr_variables[:] + [qb_target]
    if qr_ancillae:
        qubits += qr_ancillae[:or_circuit.num_ancilla_qubits]

    self.append(or_circuit.to_gate(), qubits)


QuantumCircuit.AND = logical_and
QuantumCircuit.OR = logical_or
