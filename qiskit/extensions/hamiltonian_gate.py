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

"""
Gate described by the time evolution of a Hermitian Hamiltonian operator.
"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate as NewHamiltonianGate
from qiskit.utils.deprecation import deprecate_func


class HamiltonianGate(NewHamiltonianGate):
    """Class for representing evolution by a Hamiltonian operator as a gate.

    This gate resolves to a :class:`.UnitaryGate` as :math:`U(t) = exp(-i t H)`,
    which can be decomposed into basis gates if it is 2 qubits or less, or
    simulated directly in Aer for more qubits. Note that you can also directly
    use :meth:`.QuantumCircuit.hamiltonian`.
    """

    @deprecate_func(
        since="0.45.0",
        additional_msg="This object moved to qiskit.circuit.library.HamiltonianGate.",
    )
    def __init__(self, data, time, label=None):
        """Create a gate from a hamiltonian operator and evolution time parameter t

        Args:
            data (matrix or Operator): a hermitian operator.
            time (float or ParameterExpression): time evolution parameter.
            label (str): unitary name for backend [Default: None].

        Raises:
            ExtensionError: if input data is not an N-qubit unitary operator.
        """
        super().__init__(data, time, label)


@deprecate_func(
    since="0.45.0",
    additional_msg="Instead, append a qiskit.circuit.library.HamiltonianGate to the circuit.",
)
def hamiltonian(self, operator, time, qubits, label=None):
    """Apply hamiltonian evolution to qubits.

    This gate resolves to a :class:`.UnitaryGate` as :math:`U(t) = exp(-i t H)`,
    which can be decomposed into basis gates if it is 2 qubits or less, or
    simulated directly in Aer for more qubits.

    Args:
        operator (matrix or Operator): a hermitian operator.
        time (float or ParameterExpression): time evolution parameter.
        qubits (Union[int, Tuple[int]]): The circuit qubits to apply the
            transformation to.
        label (str): unitary name for backend [Default: None].

    Returns:
        QuantumCircuit: The quantum circuit.

    Raises:
        ExtensionError: if input data is not an N-qubit unitary operator.
    """
    if not isinstance(qubits, list):
        qubits = [qubits]

    return self.append(NewHamiltonianGate(data=operator, time=time, label=label), qubits, [])


QuantumCircuit.hamiltonian = hamiltonian
