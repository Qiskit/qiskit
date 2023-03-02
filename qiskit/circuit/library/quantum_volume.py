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

"""Quantum Volume model circuit."""

from typing import Optional, Union

import numpy as np
from qiskit.quantum_info.random import random_unitary
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates.permutation import Permutation


class QuantumVolume(QuantumCircuit):
    """A quantum volume model circuit.

    The model circuits are random instances of circuits used to measure
    the Quantum Volume metric, as introduced in [1].

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    **Reference Circuit:**

    .. plot::

       from qiskit.circuit.library import QuantumVolume
       circuit = QuantumVolume(5, 6, seed=10)
       circuit.draw('mpl')

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import QuantumVolume
       from qiskit.tools.jupyter.library import _generate_circuit_library_visualization
       circuit = QuantumVolume(5, 6, seed=10, classical_permutation=False)
       _generate_circuit_library_visualization(circuit.decompose())

    **References:**

    [1] A. Cross et al. Validating quantum computers using
    randomized model circuits, Phys. Rev. A 100, 032328 (2019).
    [`arXiv:1811.12926 <https://arxiv.org/abs/1811.12926>`_]
    """

    def __init__(
        self,
        num_qubits: int,
        depth: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        classical_permutation: bool = True,
    ) -> None:
        """Create quantum volume model circuit of size num_qubits x depth.

        Args:
            num_qubits: number of active qubits in model circuit.
            depth: layers of SU(4) operations in model circuit.
            seed: Random number generator or generator seed.
            classical_permutation: use classical permutations at every layer,
                rather than quantum.
        """
        # Initialize RNG
        if seed is None:
            rng_set = np.random.default_rng()
            seed = rng_set.integers(low=1, high=1000)
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Parameters
        depth = depth or num_qubits  # how many layers of SU(4)
        width = int(np.floor(num_qubits / 2))  # how many SU(4)s fit in each layer
        name = "quantum_volume_" + str([num_qubits, depth, seed]).replace(" ", "")

        # Generator random unitary seeds in advance.
        # Note that this means we are constructing multiple new generator
        # objects from low-entropy integer seeds rather than pass the shared
        # generator object to the random_unitary function. This is done so
        # that we can use the integer seed as a label for the generated gates.
        unitary_seeds = rng.integers(low=1, high=1000, size=[depth, width])

        # For each layer, generate a permutation of qubits
        # Then generate and apply a Haar-random SU(4) to each pair
        circuit = QuantumCircuit(num_qubits, name=name)
        perm_0 = list(range(num_qubits))
        for d in range(depth):
            perm = rng.permutation(perm_0)
            if not classical_permutation:
                layer_perm = Permutation(num_qubits, perm)
                circuit.compose(layer_perm, inplace=True)
            for w in range(width):
                seed_u = unitary_seeds[d][w]
                su4 = random_unitary(4, seed=seed_u).to_instruction()
                su4.label = "su4_" + str(seed_u)
                if classical_permutation:
                    physical_qubits = int(perm[2 * w]), int(perm[2 * w + 1])
                    circuit.compose(su4, [physical_qubits[0], physical_qubits[1]], inplace=True)
                else:
                    circuit.compose(su4, [2 * w, 2 * w + 1], inplace=True)

        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_instruction(), qubits=self.qubits, inplace=True)
