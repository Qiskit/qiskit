# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary overlap circuit."""
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.exceptions import CircuitError


class UnitaryOverlap(QuantumCircuit):
    """Circuit that returns the overlap between two unitaries $V^{\dag}U|0\\rangle^{\otimes N}$.

    Input quantum circuits must represent unitary operations that are invertible.  Input
    circuits `U` and `V` will have parameters, if any, renamed `ParameterVector` `a` and `b`,
    respectively in the output circuit.

    Circuit is usually employed in computing the fidelity
    $\left|\langle 0\dots 0| V^{\dag}U|0\dots 0\\rangle\\right|^{2}$ by computing the probability
    of being in the all-zeros bit-string, or equivilently, the expectation value of projector
    $|0\dots 0\\rangle\langle 0\dots 0|$.
    """

    def __init__(self, U, V):
        """Create unitary overlap circuit

        Parameters:
            U (QuantumCircuit): Unitary acting on the ket vector
            V (QuantumCircuit): Unitary whose inverse operates on the bra vector

        Raises:
            CircuitError: Number of qubits in U and V does not match
        """
        if U.num_qubits != V.num_qubits:
            raise CircuitError(
                f"Number of qubits in unitaries does "
                f"not match: {U.num_qubits} != {V.num_qubits}."
            )
        # Vectors of new parameters, if any
        a_vec = ParameterVector("a", U.num_parameters)
        b_vec = ParameterVector("b", V.num_parameters)
        # Assign new labels so that alphabetical order matches insertion order
        circ1 = U.assign_parameters(a_vec)
        circ2 = V.assign_parameters(b_vec)
        # Generate the actual overlap circuit
        overlap = circ1.compose(circ2.inverse(), inplace=False)
        super().__init__(*overlap.qregs, name="UnitaryOverlap")
        self.compose(overlap, qubits=self.qubits, inplace=True)
