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

"""Permutation circuit."""

from typing import List, Optional

import numpy as np

from qiskit.circuit.quantumcircuit import Gate
from qiskit.circuit.exceptions import CircuitError


class Permutation(Gate):
    """An gate that permutes qubits."""

    def __init__(
        self,
        num_qubits: int,
        pattern: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Return a permutation gate.

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
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import Permutation
                import qiskit.tools.jupyter
                A = [2,4,3,0,1]
                permutation = Permutation(5, A)
                circuit = QuantumCircuit(5)
                circuit.append(permutation, [0, 1, 2, 3, 4])
                circuit.draw('mpl')

        Expanded Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import Permutation
                import qiskit.tools.jupyter
                A = [2,4,3,0,1]
                permutation = Permutation(5, A)
                circuit = QuantumCircuit(5)
                circuit.append(permutation, [0, 1, 2, 3, 4])
                %circuit_library_info circuit.decompose()
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
