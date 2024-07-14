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
from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library.generalized_gates import PermutationGate, UnitaryGate


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
       from qiskit.visualization.library import _generate_circuit_library_visualization
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
        *,
        flatten: bool = False,
    ) -> None:
        """Create quantum volume model circuit of size num_qubits x depth.

        Args:
            num_qubits: number of active qubits in model circuit.
            depth: layers of SU(4) operations in model circuit.
            seed: Random number generator or generator seed.
            classical_permutation: use classical permutations at every layer,
                rather than quantum.
            flatten: If ``False`` (the default), construct a circuit that contains a single
                instruction, which in turn has the actual volume structure.  If ``True``, construct
                the volume structure directly.
        """
        import scipy.stats

        # Parameters
        depth = depth or num_qubits  # how many layers of SU(4)
        width = num_qubits // 2  # how many SU(4)s fit in each layer
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if seed is None:
            # Get the internal entropy used to seed the default RNG, if no seed was given.  This
            # stays in the output name, so effectively stores a way of regenerating the circuit.
            # This is just best-effort only, for backwards compatibility, and isn't critical (if
            # someone needs full reproducibility, they should be manually controlling the seeding).
            seed = getattr(getattr(rng.bit_generator, "seed_seq", None), "entropy", None)

        super().__init__(
            num_qubits, name="quantum_volume_" + str([num_qubits, depth, seed]).replace(" ", "")
        )
        base = self if flatten else QuantumCircuit(num_qubits, name=self.name)

        # For each layer, generate a permutation of qubits
        # Then generate and apply a Haar-random SU(4) to each pair
        unitaries = scipy.stats.unitary_group.rvs(4, depth * width, rng).reshape(depth, width, 4, 4)
        qubits = tuple(base.qubits)
        for row in unitaries:
            perm = rng.permutation(num_qubits)
            if classical_permutation:
                for w, unitary in enumerate(row):
                    gate = UnitaryGate(unitary, check_input=False, num_qubits=2)
                    qubit = 2 * w
                    base._append(
                        CircuitInstruction(gate, (qubits[perm[qubit]], qubits[perm[qubit + 1]]))
                    )
            else:
                base._append(CircuitInstruction(PermutationGate(perm), qubits))
                for w, unitary in enumerate(row):
                    gate = UnitaryGate(unitary, check_input=False, num_qubits=2)
                    qubit = 2 * w
                    base._append(CircuitInstruction(gate, qubits[qubit : qubit + 2]))
        if not flatten:
            self._append(CircuitInstruction(base.to_instruction(), tuple(self.qubits)))
