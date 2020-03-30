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

# pylint: disable=no-member

"""Quantum Volume model circuit."""

from typing import List, Optional

import numpy as np
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class QuantumVolume(QuantumCircuit):
    """A quantum volume model circuit.

    Create quantum volume model circuit of size num_qubits x depth
    (default depth is equal to num_qubits), with a random seed.

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    Based on Cross et al. "Validating quantum computers using
    randomized model circuits", arXiv:1811.12926
    """

    def __init__(self,
                 num_qubits: int,
                 depth: Optional[int] = None,
                 seed: Optional[int] = None) -> QuantumCircuit:
        """Create a quantum volume model circuit.

        Args:
            num_qubits (int): number of active qubits in model circuit
            depth (int): layers of SU(4) operations in model circuit
            seed (int): randomization seed

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

        depth = depth or num_qubits

        circuit = QuantumCircuit(num_qubits)

        for _ in range(depth):

            # Generate uniformly random permutation Pj of [0...n-1]
            rng = np.random.RandomState(seed)
            perm = rng.permutation(width)

            # For each pair p in Pj, generate Haar random SU(4)
            for k in range(int(np.floor(width/2))):
                U = random_unitary(4, seed=seed)
                physical_qubits = int(perm[2*k]), int(perm[2*k+1])
                circuit.append(U, [physical_qubits[0], physical_qubits[1]])

        return circuit
