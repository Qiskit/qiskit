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

import inspect
from collections.abc import Callable
from typing import Any
from functools import partial
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.utils.deprecation import deprecate_arg

from .evolution_synthesis import EvolutionSynthesis


class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    :obj:`.LieTrotter` and :obj:`.SuzukiTrotter` inherit from this class.
    """

    @deprecate_arg(
        name="atomic_evolution",
        since="1.2",
        predicate=lambda callable: callable is not None
        and len(inspect.signature(callable).parameters) == 2,
        deprecation_description=(
            "The 'Callable[[Pauli | SparsePauliOp, float], QuantumCircuit]' signature of the "
            "'atomic_evolution' argument"
        ),
        additional_msg=(
            "Instead you should update your 'atomic_evolution' function to be of the following "
            "type: 'Callable[[QuantumCircuit, Pauli | SparsePauliOp, float], None]'."
        ),
        pending=True,
    )
    def __init__(
        self,
        order: int,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: (
            Callable[[Pauli | SparsePauliOp, float], QuantumCircuit]
            | Callable[[QuantumCircuit, Pauli | SparsePauliOp, float], None]
            | None
        ) = None,
        wrap: bool = False,
    ) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``,
                where all qubits are connected to one. This only takes effect when
                ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single :class:`.Pauli`, or
                :class:`.SparsePauliOp` of only commuting terms, to a circuit. The function takes in
                three arguments: the circuit to append the evolution to, the Pauli operator to
                evolve, and the evolution time. By default, a single Pauli evolution is decomposed
                into a chain of ``CX`` gates and a single ``RZ`` gate.
                Alternatively, the function can also take Pauli operator and evolution time as
                inputs and returns the circuit that will be appended to the overall circuit being
                built.
            wrap: Whether to wrap the atomic evolutions into custom gate objects. This only takes
                effect when ``atomic_evolution is None``.
        """
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers

        # user-provided atomic evolution, stored for serialization
        self._atomic_evolution = atomic_evolution
        self._cx_structure = cx_structure
        self._wrap = wrap

        # if atomic evolution is not provided, set a default
        if atomic_evolution is None:
            self.atomic_evolution = partial(
                _default_atomic_evolution, cx_structure=cx_structure, wrap=wrap
            )

        elif len(inspect.signature(atomic_evolution).parameters) == 2:

            def wrap_atomic_evolution(output, operator, time):
                definition = atomic_evolution(operator, time)
                output.compose(definition, wrap=wrap, inplace=True)

            self.atomic_evolution = wrap_atomic_evolution

        else:
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
            "wrap": self._wrap,
        }


def evolve_pauli(
    output: QuantumCircuit,
    pauli: Pauli,
    time: float | ParameterExpression = 1.0,
    cx_structure: str = "chain",
    wrap: bool = False,
    label: str | None = None,
) -> None:
    r"""Construct a circuit implementing the time evolution of a single Pauli string.

    For a Pauli string :math:`P = \{I, X, Y, Z\}^{\otimes n}` on :math:`n` qubits and an
    evolution time :math:`t`, the returned circuit implements the unitary operation

    .. math::

        U(t) = e^{-itP}.

    Since only a single Pauli string is evolved the circuit decomposition is exact.

    Args:
        output: The circuit object to which to append the evolved Pauli.
        pauli: The Pauli to evolve.
        time: The evolution time.
        cx_structure: Determine the structure of CX gates, can be either ``"chain"`` for
            next-neighbor connections or ``"fountain"`` to connect directly to the top qubit.
        wrap: Whether to wrap the single Pauli evolutions into custom gate objects.
        label: A label for the gate.
    """
    num_non_identity = len([label for label in pauli.to_label() if label != "I"])

    # first check, if the Pauli is only the identity, in which case the evolution only
    # adds a global phase
    if num_non_identity == 0:
        output.global_phase -= time
    # if we evolve on a single qubit, if yes use the corresponding qubit rotation
    elif num_non_identity == 1:
        _single_qubit_evolution(output, pauli, time, wrap)
    # same for two qubits, use Qiskit's native rotations
    elif num_non_identity == 2:
        _two_qubit_evolution(output, pauli, time, cx_structure, wrap)
    # otherwise do basis transformation and CX chains
    else:
        _multi_qubit_evolution(output, pauli, time, cx_structure, wrap)


def _single_qubit_evolution(output, pauli, time, wrap):
    dest = QuantumCircuit(1) if wrap else output
    # Note that all phases are removed from the pauli label and are only in the coefficients.
    # That's because the operators we evolved have all been translated to a SparsePauliOp.
    qubits = []
    label = ""
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        idx = 0 if wrap else i
        if pauli_i == "X":
            dest.rx(2 * time, idx)
            qubits.append(i)
            label += "X"
        elif pauli_i == "Y":
            dest.ry(2 * time, idx)
            qubits.append(i)
            label += "Y"
        elif pauli_i == "Z":
            dest.rz(2 * time, idx)
            qubits.append(i)
            label += "Z"

    if wrap:
        gate = dest.to_gate(label=f"exp(it {label})")
        qubits = [output.qubits[q] for q in qubits]
        output.append(gate, qargs=qubits, copy=False)


def _two_qubit_evolution(output, pauli, time, cx_structure, wrap):
    # Get the Paulis and the qubits they act on.
    # Note that all phases are removed from the pauli label and are only in the coefficients.
    # That's because the operators we evolved have all been translated to a SparsePauliOp.
    labels_as_array = np.array(list(reversed(pauli.to_label())))
    qubits = np.where(labels_as_array != "I")[0]
    indices = [0, 1] if wrap else qubits
    labels = np.array([labels_as_array[idx] for idx in qubits])

    dest = QuantumCircuit(2) if wrap else output

    # go through all cases we have implemented in Qiskit
    if all(labels == "X"):  # RXX
        dest.rxx(2 * time, indices[0], indices[1])
    elif all(labels == "Y"):  # RYY
        dest.ryy(2 * time, indices[0], indices[1])
    elif all(labels == "Z"):  # RZZ
        dest.rzz(2 * time, indices[0], indices[1])
    elif labels[0] == "Z" and labels[1] == "X":  # RZX
        dest.rzx(2 * time, indices[0], indices[1])
    elif labels[0] == "X" and labels[1] == "Z":  # RXZ
        dest.rzx(2 * time, indices[1], indices[0])
    else:  # all the others are not native in Qiskit, so use default the decomposition
        _multi_qubit_evolution(output, pauli, time, cx_structure, wrap)
        return

    if wrap:
        gate = dest.to_gate(label=f"exp(it {''.join(labels)})")
        qubits = [output.qubits[q] for q in qubits]
        output.append(gate, qargs=qubits, copy=False)


def _multi_qubit_evolution(output, pauli, time, cx_structure, wrap):
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
    dest = QuantumCircuit(pauli.num_qubits) if wrap else output
    dest.compose(cliff, inplace=True)
    dest.compose(chain, inplace=True)
    dest.rz(2 * time, target)
    dest.compose(chain.inverse(), inplace=True)
    dest.compose(cliff.inverse(), inplace=True)

    if wrap:
        gate = dest.to_gate(label=f"exp(it {pauli.to_label()})")
        output.append(gate, qargs=output.qubits, copy=False)


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


def _default_atomic_evolution(output, operator, time, cx_structure, wrap):
    if isinstance(operator, Pauli):
        # single Pauli operator: just exponentiate it
        evolve_pauli(output, operator, time, cx_structure, wrap)
    else:
        # sum of Pauli operators: exponentiate each term (this assumes they commute)
        pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operator.to_list()]
        for pauli, coeff in pauli_list:
            evolve_pauli(output, pauli, coeff * time, cx_structure, wrap)
