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

        # pylint: disable=cyclic-import
        from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap

        for i, j in _get_ordered_swap(pattern):
            circuit.swap(i, j)

        all_qubits = self.qubits
        self.append(circuit.to_gate(), all_qubits)
