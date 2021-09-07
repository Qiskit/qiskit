# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A gate to implement time-evolution of a single Pauli string."""

from typing import Union, Optional
from qiskit.quantum_info import Pauli
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit


class PauliEvolutionGate(Gate):
    """Time-evolution of a single Pauli string."""

    def __init__(
        self,
        pauli: Pauli,
        time: Union[float, ParameterExpression] = 1.0,
        label: Optional[str] = None,
        cx_structure: str = "chain",
    ) -> None:
        """
        Args:
            operator: The Pauli to evolve.
        """
        super().__init__(
            name=f"exp(it {pauli.to_label()})", num_qubits=pauli.num_qubits, params=[], label=label
        )

        self.time = time
        self.pauli = pauli
        self.cx_structure = cx_structure

    def _define(self):
        # first check if we evolve on a single qubit, if yes use the corresponding qubit rotation
        if len([label for label in self.pauli.to_label() if label != 'I']) <= 1:
            self.definition = self._single_qubit_evolution()
        else:
            self.definition = self._multi_qubit_evolution()

    def _single_qubit_evolution(self):
        definition = QuantumCircuit(self.pauli.num_qubits)
        for i, pauli_i in enumerate(reversed(self.pauli.to_label())):
            if pauli_i == 'X':
                definition.rx(2 * self.time, i)
            elif pauli_i == 'Y':
                definition.ry(2 * self.time, i)
            elif pauli_i == 'Z':
                definition.rz(2 * self.time, i)

        return definition

    def _multi_qubit_evolution(self):
        # get diagonalizing clifford
        cliff = diagonalizing_clifford(self.pauli)

        # get CX chain to reduce the evolution to the top qubit
        if self.cx_structure == "chain":
            chain = cnot_chain(self.pauli)
        else:
            chain = cnot_fountain(self.pauli)

        # determine qubit to do the rotation on
        target = None
        for i, pauli_i in enumerate(reversed(self.pauli.to_label())):
            if pauli_i != 'I':
                target = i
                break

        # build the evolution as: diagonalization, reduction, 1q evolution, followed by inverses
        definition = QuantumCircuit(self.pauli.num_qubits)
        definition.compose(cliff, inplace=True)
        definition.compose(chain, inplace=True)
        definition.rz(2 * self.time, target)
        definition.compose(chain.inverse(), inplace=True)
        definition.compose(cliff.inverse(), inplace=True)

        return definition

    def inverse(self) -> "PauliEvolutionGate":
        return PauliEvolutionGate(pauli=self.pauli, time=-self.time)


def diagonalizing_clifford(pauli: Pauli) -> QuantumCircuit:
    """Get the clifford circuit to diagonalize the Pauli operator.

    Args:
        pauli: The Pauli to diagonalize.

    Returns:
        A circuit to diagonalize.
    """
    cliff = QuantumCircuit(pauli.num_qubits)
    for i, pauli_i in enumerate(pauli.to_label()):
        if pauli_i == 'Y':
            cliff.sdg(i)
        if pauli_i in ['X', 'Y']:
            cliff.h(i)

    return cliff


def cnot_chain(pauli: Pauli) -> QuantumCircuit:
    """CX chain.

    For example, for the Pauli with the label 'XYZIX'.

    .. code-block::

                       ┌───┐
        q_0: ──────────┤ X ├
                       └─┬─┘
        q_1: ────────────┼──
                  ┌───┐  │
        q_2: ─────┤ X ├──■──
             ┌───┐└─┬─┘
        q_3: ┤ X ├──■───────
             └─┬─┘
        q_4: ──■────────────

    Args:
        pauli: The Pauli for which to construct the CX chain.

    Returns:
        A circuit implementing the CX chain.
    """

    chain = QuantumCircuit(pauli.num_qubits)
    control, target = None, None

    # iterate over the Pauli's and add CNOTs
    for i, pauli_i in enumerate(pauli.to_label()):
        i = pauli.num_qubits - i - 1
        if pauli_i != 'I':
            if control is None:
                control = i
            else:
                target = i

        if control is not None and target is not None:
            chain.cx(control, target)
            control = i
            target = None

    return chain


def cnot_fountain(pauli: Pauli) -> QuantumCircuit:
    """CX chain in the fountain shape.

    For example, for the Pauli with the label 'XYZIX'.

    .. code-block::

             ┌───┐┌───┐┌───┐
        q_0: ┤ X ├┤ X ├┤ X ├
             └─┬─┘└─┬─┘└─┬─┘
        q_1: ──┼────┼────┼──
               │    │    │
        q_2: ──■────┼────┼──
                    │    │
        q_3: ───────■────┼──
                         │
        q_4: ────────────■──

    Args:
        pauli: The Pauli for which to construct the CX chain.

    Returns:
        A circuit implementing the CX chain.
    """

    chain = QuantumCircuit(pauli.num_qubits)
    control, target = None, None
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i != 'I':
            if target is None:
                target = i
            else:
                control = i

        if control is not None and target is not None:
            chain.cx(control, target)
            control = None

    return chain
