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

# The structure of the code is based on Emanuel Malvetti's semester thesis at
# ETH in 2018, which was supervised by Raban Iten and Prof. Renato Renner.

# pylint: disable=invalid-name
# pylint: disable=missing-param-doc
# pylint: disable=missing-type-doc

"""
Uniformly controlled gates (also called multiplexed gates).

These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state |i> (in the computational basis),
a single-qubit unitary U_i is applied to the target qubit.

This gate is represented by a block-diagonal matrix, where each block is a
2x2 unitary:

    [[U_0, 0,   ....,        0],
     [0,   U_1, ....,        0],
                .
                    .
     [0,   0,  ...., U_(2^k-1)]]
"""

import math

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates.uc import UCGate as NewUCGate
from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_func


class UCGate(NewUCGate):
    """Uniformly controlled gate (also called multiplexed gate).
    The decomposition is based on: https://arxiv.org/pdf/quant-ph/0410066.pdf.
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.UCGate."
    )
    def __init__(self, gate_list, up_to_diagonal=False):
        """UCGate Gate initializer.

        Args:
            gate_list (list[ndarray]): list of two qubit unitaries [U_0,...,U_{2^k-1}],
                where each single-qubit unitary U_i is given as a 2*2 numpy array.

            up_to_diagonal (bool): determines if the gate is implemented up to a diagonal.
                or if it is decomposed completely (default: False).
                If the UCGate u is decomposed up to a diagonal d, this means that the circuit
                implements a unitary u' such that d.u'=u.

        Raises:
            QiskitError: in case of bad input to the constructor
        """
        super().__init__(gate_list, up_to_diagonal)


@deprecate_func(
    since="0.45.0", additional_msg="Instead, append a qiskit.circuit.library.UCGate to the circuit."
)
def uc(self, gate_list, q_controls, q_target, up_to_diagonal=False):
    """Attach a uniformly controlled gates (also called multiplexed gates) to a circuit.

    The decomposition was introduced by Bergholm et al. in
    https://arxiv.org/pdf/quant-ph/0410066.pdf.

    Args:
        gate_list (list[ndarray]): list of two qubit unitaries [U_0,...,U_{2^k-1}],
            where each single-qubit unitary U_i is a given as a 2*2 array
        q_controls (QuantumRegister|list[(QuantumRegister,int)]): list of k control qubits.
            The qubits are ordered according to their significance in the computational basis.
            For example if q_controls=[q[1],q[2]] (with q = QuantumRegister(2)),
            the unitary U_0 is performed if q[1] and q[2] are in the state zero, U_1 is
            performed if q[2] is in the state zero and q[1] is in the state one, and so on
        q_target (QuantumRegister|(QuantumRegister,int)):  target qubit, where we act on with
            the single-qubit gates.
        up_to_diagonal (bool): If set to True, the uniformly controlled gate is decomposed up
            to a diagonal gate, i.e. a unitary u' is implemented such that there exists a
            diagonal gate d with u = d.dot(u'), where the unitary u describes the uniformly
            controlled gate

    Returns:
        QuantumCircuit: the uniformly controlled gate is attached to the circuit.

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
            raise QiskitError(
                "The target qubit is a QuantumRegister containing more than one qubit."
            )
    # Check if q_controls has type "list"
    if not isinstance(q_controls, list):
        raise QiskitError(
            "The control qubits must be provided as a list"
            " (also if there is only one control qubit)."
        )
    # Check if gate_list has type "list"
    if not isinstance(gate_list, list):
        raise QiskitError("The single-qubit unitaries are not provided in a list.")
        # Check if number of gates in gate_list is a positive power of two
    num_contr = math.log2(len(gate_list))
    if num_contr < 0 or not num_contr.is_integer():
        raise QiskitError(
            "The number of controlled single-qubit gates is not a non negative power of 2."
        )
    # Check if number of control qubits does correspond to the number of single-qubit rotations
    if num_contr != len(q_controls):
        raise QiskitError(
            "Number of controlled gates does not correspond to the number of control qubits."
        )
    return self.append(UCGate(gate_list, up_to_diagonal), [q_target] + q_controls)


QuantumCircuit.uc = uc
