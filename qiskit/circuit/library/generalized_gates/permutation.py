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

"""Permutation circuit (the old way to specify permutations, which is required for
backward compatibility and which will be eventually deprecated) and the permutation
gate (the new way to specify permutations, allowing a variety of synthesis algorithms).
"""

from __future__ import annotations

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumcircuit import Gate
from qiskit.circuit.exceptions import CircuitError


class Permutation(QuantumCircuit):
    """An n_qubit circuit that permutes qubits."""

    def __init__(
        self,
        num_qubits: int,
        pattern: list[int] | np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        """Return an n_qubit permutation circuit implemented using SWAPs.

        Args:
            num_qubits: circuit width.
            pattern: permutation pattern, describing which qubits occupy the
                positions 0, 1, 2, etc. after applying the permutation, that
                is ``pattern[k] = m`` when the permutation maps qubit ``m``
                to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
                means that qubit ``2`` goes to position ``0``, qubit ``4``
                goes to the position ``1``, etc. The pattern can also be ``None``,
                in which case a random permutation over ``num_qubits`` is
                created.
            seed: random seed in case a random permutation is requested.

        Raises:
            CircuitError: if permutation pattern is malformed.

        Reference Circuit:
            .. plot::

               from qiskit.circuit.library import Permutation
               A = [2,4,3,0,1]
               circuit = Permutation(5, A)
               circuit.draw('mpl')

        Expanded Circuit:
            .. plot::

               from qiskit.circuit.library import Permutation
               from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
               A = [2,4,3,0,1]
               circuit = Permutation(5, A)
               _generate_circuit_library_visualization(circuit.decompose())
        """
        if pattern is not None:
            if sorted(pattern) != list(range(num_qubits)):
                raise CircuitError(
                    "Permutation pattern must be some ordering of 0..num_qubits-1 in a list."
                )
            pattern = np.array(pattern)
        else:
            rng = np.random.default_rng(seed)
            pattern = np.arange(num_qubits)
            rng.shuffle(pattern)

        name = "permutation_" + np.array_str(pattern).replace(" ", ",")

        circuit = QuantumCircuit(num_qubits, name=name)

        super().__init__(num_qubits, name=name)

        # pylint: disable=cyclic-import
        from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap

        for i, j in _get_ordered_swap(pattern):
            circuit.swap(i, j)

        all_qubits = self.qubits
        self.append(circuit.to_gate(), all_qubits)


class PermutationGate(Gate):
    """A gate that permutes qubits."""

    def __init__(
        self,
        pattern: list[int],
    ) -> None:
        """Return a permutation gate.

        Args:
            pattern: permutation pattern, describing which qubits occupy the
                positions 0, 1, 2, etc. after applying the permutation, that
                is ``pattern[k] = m`` when the permutation maps qubit ``m``
                to position ``k``. As an example, the pattern ``[2, 4, 3, 0, 1]``
                means that qubit ``2`` goes to position ``0``, qubit ``4``
                goes to the position ``1``, etc.

        Raises:
            CircuitError: if permutation pattern is malformed.

        Reference Circuit:
            .. plot::

                from qiskit.circuit.quantumcircuit import QuantumCircuit
                from qiskit.circuit.library import PermutationGate
                A = [2,4,3,0,1]
                permutation = PermutationGate(A)
                circuit = QuantumCircuit(5)
                circuit.append(permutation, [0, 1, 2, 3, 4])
                circuit.draw('mpl')

        Expanded Circuit:
            .. plot::

                from qiskit.circuit.quantumcircuit import QuantumCircuit
                from qiskit.circuit.library import PermutationGate
                from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
                A = [2,4,3,0,1]
                permutation = PermutationGate(A)
                circuit = QuantumCircuit(5)
                circuit.append(permutation, [0, 1, 2, 3, 4])

                _generate_circuit_library_visualization(circuit.decompose())
        """
        num_qubits = len(pattern)
        if sorted(pattern) != list(range(num_qubits)):
            raise CircuitError(
                "Permutation pattern must be some ordering of 0..num_qubits-1 in a list."
            )
        pattern = np.array(pattern)

        super().__init__(name="permutation", num_qubits=num_qubits, params=[pattern])

    def __array__(self, dtype=None):
        """Return a numpy.array for the Permutation gate."""
        nq = len(self.pattern)
        mat = np.zeros((2**nq, 2**nq), dtype=dtype)

        for r in range(2**nq):
            # convert row to bitstring, reverse, apply permutation pattern, reverse again,
            # and convert to row
            bit = bin(r)[2:].zfill(nq)[::-1]
            permuted_bit = "".join([bit[j] for j in self.pattern])
            pr = int(permuted_bit[::-1], 2)
            mat[pr, r] = 1

        return mat

    def validate_parameter(self, parameter):
        """Parameter validation."""
        return parameter

    @property
    def pattern(self):
        """Returns the permutation pattern defining this permutation."""
        return self.params[0]

    def inverse(self):
        """Returns the inverse of the permutation."""

        # pylint: disable=cyclic-import
        from qiskit.synthesis.permutation.permutation_utils import _inverse_pattern

        return PermutationGate(pattern=_inverse_pattern(self.pattern))

    def _qasm2_decomposition(self):
        # pylint: disable=cyclic-import
        from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap

        name = f"permutation__{'_'.join(str(n) for n in self.pattern)}_"
        out = QuantumCircuit(self.num_qubits, name=name)
        for i, j in _get_ordered_swap(self.pattern):
            out.swap(i, j)
        return out.to_gate()
