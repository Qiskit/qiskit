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

"""A product formula base for decomposing non-commuting operator exponentials."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from functools import partial
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

from .evolution_synthesis import EvolutionSynthesis


class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    :obj:`.LieTrotter` and :obj:`.SuzukiTrotter` inherit from this class.
    """

    def __init__(
        self,
        order: int,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: Callable[[Pauli | SparsePauliOp, float], QuantumCircuit] | None = None,
    ) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``,
                where all qubits are connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single
                Pauli string. Per default, a single Pauli evolution is decomposed in a CX chain
                and a single qubit Z rotation.
        """
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers

        # user-provided atomic evolution, stored for serialization
        self._atomic_evolution = atomic_evolution
        self._cx_structure = cx_structure

        # if atomic evolution is not provided, set a default
        if atomic_evolution is None:
            atomic_evolution = partial(_default_atomic_evolution, cx_structure=cx_structure)

        self.atomic_evolution = atomic_evolution

    @property
    def settings(self) -> dict[str, Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: If a custom atomic evolution is set, which cannot be serialized.
        """
        if self._atomic_evolution is not None:
            raise NotImplementedError(
                "Cannot serialize a product formula with a custom atomic evolution."
            )

        return {
            "order": self.order,
            "reps": self.reps,
            "insert_barriers": self.insert_barriers,
            "cx_structure": self._cx_structure,
        }


def evolve_pauli(
    pauli: Pauli,
    time: float | ParameterExpression = 1.0,
    cx_structure: str = "chain",
    label: str | None = None,
    output: QuantumCircuit | None = None,
) -> QuantumCircuit | None:
    r"""Construct a circuit implementing the time evolution of a single Pauli string.

    For a Pauli string :math:`P = \{I, X, Y, Z\}^{\otimes n}` on :math:`n` qubits and an
    evolution time :math:`t`, the returned circuit implements the unitary operation

    .. math::

        U(t) = e^{-itP}.

    Since only a single Pauli string is evolved the circuit decomposition is exact.

    Args:
        pauli: The Pauli to evolve.
        time: The evolution time.
        cx_structure: Determine the structure of CX gates, can be either ``"chain"`` for
            next-neighbor connections or ``"fountain"`` to connect directly to the top qubit.
        label: A label for the gate.
        output: An optional existing circuit to which to append the evolved Pauli. When a circuit is
            provided, ``None`` will be returned.

    Returns:
        A quantum circuit implementing the time evolution of the Pauli. If ``output is not None``
        then ``None`` will be returned since the evolved Pauli will have been appended to the
        ``output`` circuit.
    """
    num_non_identity = len([label for label in pauli.to_label() if label != "I"])

    # first check, if the Pauli is only the identity, in which case the evolution only
    # adds a global phase
    if num_non_identity == 0:
        if output is None:
            definition = QuantumCircuit(pauli.num_qubits, global_phase=-time)
        else:
            output.global_phase -= time
    # if we evolve on a single qubit, if yes use the corresponding qubit rotation
    elif num_non_identity == 1:
        definition = _single_qubit_evolution(pauli, time, output)
    # same for two qubits, use Qiskit's native rotations
    elif num_non_identity == 2:
        definition = _two_qubit_evolution(pauli, time, cx_structure, output)
    # otherwise do basis transformation and CX chains
    else:
        definition = _multi_qubit_evolution(pauli, time, cx_structure, output)

    if output is not None:
        return None

    definition.name = f"exp(it {pauli.to_label()})"

    return definition


def _single_qubit_evolution(pauli, time, output):
    if output is None:
        dest = QuantumCircuit(pauli.num_qubits)
    else:
        dest = output
    # Note that all phases are removed from the pauli label and are only in the coefficients.
    # That's because the operators we evolved have all been translated to a SparsePauliOp.
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i == "X":
            dest.rx(2 * time, i)
        elif pauli_i == "Y":
            dest.ry(2 * time, i)
        elif pauli_i == "Z":
            dest.rz(2 * time, i)

    return dest if output is None else None


def _two_qubit_evolution(pauli, time, cx_structure, output):
    # Get the Paulis and the qubits they act on.
    # Note that all phases are removed from the pauli label and are only in the coefficients.
    # That's because the operators we evolved have all been translated to a SparsePauliOp.
    labels_as_array = np.array(list(reversed(pauli.to_label())))
    qubits = np.where(labels_as_array != "I")[0]
    labels = np.array([labels_as_array[idx] for idx in qubits])

    if output is None:
        dest = QuantumCircuit(pauli.num_qubits)
    else:
        dest = output

    # go through all cases we have implemented in Qiskit
    if all(labels == "X"):  # RXX
        dest.rxx(2 * time, qubits[0], qubits[1])
    elif all(labels == "Y"):  # RYY
        dest.ryy(2 * time, qubits[0], qubits[1])
    elif all(labels == "Z"):  # RZZ
        dest.rzz(2 * time, qubits[0], qubits[1])
    elif labels[0] == "Z" and labels[1] == "X":  # RZX
        dest.rzx(2 * time, qubits[0], qubits[1])
    elif labels[0] == "X" and labels[1] == "Z":  # RXZ
        dest.rzx(2 * time, qubits[1], qubits[0])
    else:  # all the others are not native in Qiskit, so use default the decomposition
        dest = _multi_qubit_evolution(pauli, time, cx_structure, output)

    return dest if output is None else None


def _multi_qubit_evolution(pauli, time, cx_structure, output):
    # get diagonalizing clifford
    cliff = diagonalizing_clifford(pauli)

    # get CX chain to reduce the evolution to the top qubit
    if cx_structure == "chain":
        chain = cnot_chain(pauli)
    else:
        chain = cnot_fountain(pauli)

    # determine qubit to do the rotation on
    target = None
    # Note that all phases are removed from the pauli label and are only in the coefficients.
    # That's because the operators we evolved have all been translated to a SparsePauliOp.
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i != "I":
            target = i
            break

    # build the evolution as: diagonalization, reduction, 1q evolution, followed by inverses
    if output is None:
        dest = QuantumCircuit(pauli.num_qubits)
    else:
        dest = output
    dest.compose(cliff, inplace=True)
    dest.compose(chain, inplace=True)
    dest.rz(2 * time, target)
    dest.compose(chain.inverse(), inplace=True)
    dest.compose(cliff.inverse(), inplace=True)

    return dest if output is None else None


def diagonalizing_clifford(pauli: Pauli) -> QuantumCircuit:
    """Get the clifford circuit to diagonalize the Pauli operator.

    Args:
        pauli: The Pauli to diagonalize.

    Returns:
        A circuit to diagonalize.
    """
    cliff = QuantumCircuit(pauli.num_qubits)
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i == "Y":
            cliff.sdg(i)
        if pauli_i in ["X", "Y"]:
            cliff.h(i)

    return cliff


def cnot_chain(pauli: Pauli) -> QuantumCircuit:
    """CX chain.

    For example, for the Pauli with the label 'XYZIX'.

    .. parsed-literal::

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
        if pauli_i != "I":
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

    .. parsed-literal::

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
        if pauli_i != "I":
            if target is None:
                target = i
            else:
                control = i

        if control is not None and target is not None:
            chain.cx(control, target)
            control = None

    return chain


def _default_atomic_evolution(operator, time, cx_structure):
    if isinstance(operator, Pauli):
        # single Pauli operator: just exponentiate it
        evolution_circuit = evolve_pauli(operator, time, cx_structure)
    else:
        # sum of Pauli operators: exponentiate each term (this assumes they commute)
        pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operator.to_list()]
        name = f"exp(it {[pauli.to_label() for pauli, _ in pauli_list]})"
        evolution_circuit = QuantumCircuit(operator.num_qubits, name=name)
        for pauli, coeff in pauli_list:
            evolve_pauli(pauli, coeff * time, cx_structure, output=evolution_circuit)

    return evolution_circuit
