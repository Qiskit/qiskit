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

# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-param-doc
# pylint: disable=missing-type-doc

"""
Generic isometries from m to n qubits.
"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.generalized_gates.isometry import Isometry as NewIsometry
from qiskit.utils.deprecation import deprecate_func

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Isometry(NewIsometry):
    """
    Decomposition of arbitrary isometries from m to n qubits. In particular, this allows to
    decompose unitaries (m=n) and to do state preparation (m=0).

    The decomposition is based on https://arxiv.org/abs/1501.06911.

    Args:
        isometry (ndarray): an isometry from m to n qubits, i.e., a (complex)
            np.ndarray of dimension 2^n*2^m with orthonormal columns (given
            in the computational basis specified by the order of the ancillas
            and the input qubits, where the ancillas are considered to be more
            significant than the input qubits).

        num_ancillas_zero (int): number of additional ancillas that start in the state ket(0)
            (the n-m ancillas required for providing the output of the isometry are
            not accounted for here).

        num_ancillas_dirty (int): number of additional ancillas that start in an arbitrary state

        epsilon (float) (optional): error tolerance of calculations
    """

    @deprecate_func(
        since="0.45.0", additional_msg="This object moved to qiskit.circuit.library.Isometry."
    )
    def __init__(self, isometry, num_ancillas_zero, num_ancillas_dirty, epsilon=_EPS):
        super().__init__(isometry, num_ancillas_zero, num_ancillas_dirty, epsilon)


@deprecate_func(
    since="0.45.0",
    additional_msg="Instead, append a qiskit.circuit.library.Isometry to the circuit.",
)
def iso(
    self,
    isometry,
    q_input,
    q_ancillas_for_output,
    q_ancillas_zero=None,
    q_ancillas_dirty=None,
    epsilon=_EPS,
):
    """
    Attach an arbitrary isometry from m to n qubits to a circuit. In particular,
    this allows to attach arbitrary unitaries on n qubits (m=n) or to prepare any state
    on n qubits (m=0).
    The decomposition used here was introduced by Iten et al. in https://arxiv.org/abs/1501.06911.

    Args:
        isometry (ndarray): an isometry from m to n qubits, i.e., a (complex) ndarray of
            dimension 2^n×2^m with orthonormal columns (given in the computational basis
            specified by the order of the ancillas and the input qubits, where the ancillas
            are considered to be more significant than the input qubits.).
        q_input (QuantumRegister|list[Qubit]): list of m qubits where the input
            to the isometry is fed in (empty list for state preparation).
        q_ancillas_for_output (QuantumRegister|list[Qubit]): list of n-m ancilla
            qubits that are used for the output of the isometry and which are assumed to start
            in the zero state. The qubits are listed with increasing significance.
        q_ancillas_zero (QuantumRegister|list[Qubit]): list of ancilla qubits
            which are assumed to start in the zero state. Default is q_ancillas_zero = None.
        q_ancillas_dirty (QuantumRegister|list[Qubit]): list of ancilla qubits
            which can start in an arbitrary state. Default is q_ancillas_dirty = None.
        epsilon (float): error tolerance of calculations.
            Default is epsilon = _EPS.

    Returns:
        QuantumCircuit: the isometry is attached to the quantum circuit.

    Raises:
        QiskitError: if the array is not an isometry of the correct size corresponding to
            the provided number of qubits.
    """
    if q_input is None:
        q_input = []
    if q_ancillas_for_output is None:
        q_ancillas_for_output = []
    if q_ancillas_zero is None:
        q_ancillas_zero = []
    if q_ancillas_dirty is None:
        q_ancillas_dirty = []

    if isinstance(q_input, QuantumRegister):
        q_input = q_input[:]
    if isinstance(q_ancillas_for_output, QuantumRegister):
        q_ancillas_for_output = q_ancillas_for_output[:]
    if isinstance(q_ancillas_zero, QuantumRegister):
        q_ancillas_zero = q_ancillas_zero[:]
    if isinstance(q_ancillas_dirty, QuantumRegister):
        q_ancillas_dirty = q_ancillas_dirty[:]

    return self.append(
        NewIsometry(isometry, len(q_ancillas_zero), len(q_ancillas_dirty), epsilon=epsilon),
        q_input + q_ancillas_for_output + q_ancillas_zero + q_ancillas_dirty,
    )


# support both QuantumCircuit.iso and QuantumCircuit.isometry
QuantumCircuit.iso = iso
QuantumCircuit.isometry = iso
