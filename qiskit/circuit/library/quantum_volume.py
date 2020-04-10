# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Quantum Volume model circuit."""

from typing import Optional

import numpy as np
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit import QuantumCircuit


class QuantumVolume(QuantumCircuit):
    """A quantum volume model circuit.

    The model circuits are random instances of circuits used to measure
    the Quantum Volume metric, as introduced in [1].

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    **References:**

    [1] A. Cross et al. Validating quantum computers using
    randomized model circuits, 2018.
    `arXiv:1811.12926 <https://arxiv.org/abs/1811.12926>`_
    """

    def __init__(self,
                 num_qubits: int,
                 depth: Optional[int] = None,
                 seed: Optional[int] = None) -> QuantumCircuit:
        """Create quantum volume model circuit of size num_qubits x depth.

        Args:
            num_qubits: number of active qubits in model circuit.
            depth: layers of SU(4) operations in model circuit.
            seed: randomization seed.

        Returns:
            QuantumCircuit: a randomly constructed quantum volume model circuit.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import QuantumVolume
                import qiskit.tools.jupyter
                circuit = QuantumVolume(5, seed=42)
                %circuit_library_info circuit
        """
        super().__init__(num_qubits, name="volume")

        depth = depth or num_qubits  # how many layers of SU(4)
        width = int(np.floor(num_qubits/2))  # how many SU(4)s fit in each layer
        rng = np.random.RandomState(seed)

        unitary_seeds = rng.randint(low=1, high=1000, size=[depth, width])

        # For each layer, generate a permutation of qubits
        # Then generate and apply a Haar-random SU(4) to each pair
        perm_0 = list(range(num_qubits))
        for d in range(depth):
            perm = rng.permutation(perm_0)
            for w in range(width):
                physical_qubits = int(perm[2*w]), int(perm[2*w+1])
                su4 = random_unitary(4, seed=unitary_seeds[d][w])
                self.append(su4, [physical_qubits[0], physical_qubits[1]])
