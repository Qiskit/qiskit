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

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class Permutation(QuantumCircuit):
    """An n_qubit circuit that permutes qubits."""

    def __init__(
        self,
        num_qubits: int,
        pattern: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Return an n_qubit permutation circuit implemented using SWAPs.

        Args:
            num_qubits: circuit width.
            pattern: permutation pattern. If None, permute randomly.
            seed: random seed in case a random permutation is requested.

        Raises:
            CircuitError: if permutation pattern is malformed.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import Permutation
                import qiskit.tools.jupyter
                A = [2,4,3,0,1]
                circuit = Permutation(5, A)
                circuit.draw('mpl')

        Expanded Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import Permutation
                import qiskit.tools.jupyter
                A = [2,4,3,0,1]
                circuit = Permutation(5, A)
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

        name = "permutation_" + np.array_str(pattern).replace(" ", ",")

        circuit = QuantumCircuit(num_qubits, name=name)

        super().__init__(num_qubits, name=name)
        for i, j in _get_ordered_swap(pattern):
            circuit.swap(i, j)

        all_qubits = self.qubits
        self.append(circuit.to_gate(), all_qubits)


def _get_ordered_swap(permutation_in):
    """This attempts to sort the input permutation by iterating through the
    permutation list and swapping the element with where the actual index occurs and
    and tracking the swaps.
    """
    permutation = list(permutation_in[:])
    swap_list = []
    for i, val in enumerate(permutation):
        if val != i:
            j = permutation.index(i)
            swap_list.append((i, j))
            permutation[i], permutation[j] = permutation[j], permutation[i]
    swap_list.reverse()
    return swap_list
