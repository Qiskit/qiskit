# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Swap network."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Callable, Optional

from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit


class SwapNetwork(QuantumCircuit):
    """A swap network circuit.

    A swap network applies arbitrary pairwise interactions between qubits
    using only linear connectivity. It works by reversing the order of qubits
    with a sequence of swap gates and applying an operation when the relevant
    qubits become adjacent. Note that at the end of the operation
    the qubit ordering has been reversed; this must be kept track of.

    Reference: `arXiv:1711.04789`_

    .. _arXiv:1711.04789: https://arxiv.org/abs/1711.04789

    Examples:

        .. jupyter-execute::

            from qiskit.circuit.library import SwapNetwork
            swap_network = SwapNetwork(5)
            swap_network.draw()

        .. jupyter-execute::

            from qiskit.circuit.library import SwapNetwork
            swap_network = SwapNetwork(5, offset=True)
            swap_network.draw()

        .. jupyter-execute::

            from qiskit.circuit.library import CZGate, SwapNetwork
            swap_network = SwapNetwork(
                4,
                operation=lambda i, j: CZGate()
            )
            swap_network.draw()
    """

    def __init__(
        self,
        n_qubits: int,
        operation: Optional[Callable[[int, int], Instruction]] = None,
        swap_gate: Optional[Gate] = None,
        offset: bool = False,
        register_name: Optional[str] = None,
        **circuit_kwargs,
    ) -> None:
        r"""
        Args:
            n_qubits: The number of qubits.
            operation: Returns interactions to perform between qubits as
                they are swapped past each other. A call to this function takes the
                form ``operation(i, j)`` where ``i`` and ``j`` are indices
                representing the "logical" qubits as they were initially ordered,
                It returns the instruction to perform on the physical qubits
                currently storing logical qubits ``i`` and ``j`` (in that order).
            swap_gate: The swap gate to use. Defaults to the normal SWAP gate (an instance
                of :class:`~.SwapGate`).
            offset: If True, then qubit 0 will participate in odd-numbered layers
                instead of even-numbered layers.
            register_name: The name to use for the quantum register.
        """
        register = QuantumRegister(n_qubits, name=register_name)
        super().__init__(register, **circuit_kwargs)
        for gate, qubits in _swap_network(register, operation, swap_gate, offset):
            self.append(gate, qubits)


def _swap_network(
    register: QuantumRegister,
    operation: Optional[Callable[[int, int], Instruction]],
    swap_gate: Optional[Gate],
    offset: bool = False,
) -> Iterator[tuple[Instruction, tuple[Qubit, ...]]]:
    swap_gate = swap_gate or SwapGate()
    n_qubits = len(register)
    order = list(range(n_qubits))
    for layer_num in range(n_qubits):
        lowest_active_qubit = (layer_num + offset) % 2
        active_pairs = ((i, i + 1) for i in range(lowest_active_qubit, n_qubits - 1, 2))
        for a, b in active_pairs:
            i, j = order[a], order[b]
            i_qubit, j_qubit = register[a], register[b]
            if operation:
                yield operation(i, j), (i_qubit, j_qubit)
            yield swap_gate, (i_qubit, j_qubit)
            order[a], order[b] = order[b], order[a]
