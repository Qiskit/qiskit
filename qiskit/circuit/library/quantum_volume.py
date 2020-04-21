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

MAX_INT = np.iinfo(np.int32).max-1


class QuantumVolume(QuantumCircuit):
    """A model quantum volume circuit [1].

    The model circuits are random instances of circuits
    consisting of layers of Haar random elements of SU(4)
    applied between corresponding pairs of qubits in a random
    bipartition.

    If depth == num_qubits (default), the circuit name includes
    the Quantum Volume that the circuit corresponds to else it is
    the width x depth of the circuit, the seed used in the random
    number generator, and the offset from this seed (here always
    equal to zero.). This completely specifies the circuit.

    **References:**

    [1] A. Cross et al. Validating quantum computers using
    randomized model circuits, 2018.
    `arXiv:1811.12926 <https://arxiv.org/abs/1811.12926>`_
    """

    def __init__(self,
                 num_qubits: int,
                 depth: Optional[int] = None,
                 seed: Optional[int] = None) -> QuantumCircuit:
        """Create a model quantum volume circuit of
        size num_qubits x depth.

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
        depth = depth or num_qubits  # how many layers of SU(4)
        width = int(np.floor(num_qubits/2))  # how many SU(4)s fit in each layer
        if seed is None:
            seed = np.random.randint(MAX_INT)

        if depth == num_qubits:
            name = 'QV{}_{}+{}'.format(2**num_qubits, seed, 0)
        else:
            name = "genQV_{}x{}_{}+{}".format(num_qubits, depth, seed, 0)
        super().__init__(num_qubits, name=name)

        rng = np.random.RandomState(seed)

        # For each layer, generate a permutation of qubits
        # Then generate and apply a Haar-random SU(4) to each pair
        perm_0 = list(range(num_qubits))
        for _ in range(depth):
            perm = rng.permutation(perm_0)
            for w in range(width):
                physical_qubits = int(perm[2*w]), int(perm[2*w+1])
                su4 = random_unitary(4, seed=rng.randint(MAX_INT))
                self.append(su4, [physical_qubits[0], physical_qubits[1]])


class QuantumVolumeGenerator():
    """A generator for quantum volume circuits.

    If depth == num_qubits (default), the circuit names include
    the Quantum Volume that the circuit corresponds to else it is
    the width x depth of the circuit, the seed used in the random
    number generator, and the offset from this seed, i.e. the
    number of times the random number generator has been called.
    """

    def __init__(self, num_qubits: int,
                 depth: Optional[int] = None,
                 seed: Optional[int] = None):
        """A generator for quantum volume circuits.

        Parameters:
            num_qubits: Number of qubits in QV circuit.
            depth: Number of layers of random pairwise unitaries.
            seed: Optional seed at which to start generator

        Example:
        .. jupyter-execute::

            from qiskit.circuit.library import QuantumVolumeGenerator
            qv_gen = QuantumVolumeGenerator(4, seed=9876)
            qv16_circs = qv_gen(5)
            for circ in qv16_circs:
                print(circ.name)
        """
        if seed is None:
            seed = np.random.randint(MAX_INT)
        self.seed = seed
        self.rnd = np.random.RandomState(self.seed) # pylint: disable=no-member
        self.num_qubits = num_qubits
        self.depth = depth or num_qubits
        if self.num_qubits == self.depth:
            self.circ_name = 'QV{}_{}'.format(2**self.num_qubits,
                                              self.seed)
        else:
            self.circ_name = "genQV_{}x{}_{}".format(num_qubits, depth, seed)
        self.count = 0

    def __call__(self, samples: Optional[int] = None):
        """Creates a collection of Quantum Volume circuits.

        Parameters:
            samples: Number of circuits to generate.

        Returns:
            list: A list of QuantumCircuits.
        """
        if samples is None:
            samples = 1
        out = []
        for _ in range(samples):
            qc_name = self.circ_name + '+{}'.format(self.count)
            qc = QuantumCircuit(self.num_qubits, name=qc_name)
            for _ in range(self.depth):
                # Generate uniformly random permutation Pj of [0...n-1]
                perm = self.rnd.permutation(self.num_qubits)
                # For each pair p in Pj, generate Haar random SU(4)
                for k in range(int(self.num_qubits/2)):
                    su4 = random_unitary(4, seed=self.rnd.randint(MAX_INT))
                    pair = int(perm[2*k]), int(perm[2*k+1])
                    qc.append(su4, [pair[0], pair[1]])
            out.append(qc)
            self.count += 1
        return out

    def __next__(self):
        return self.__call__()[0]
