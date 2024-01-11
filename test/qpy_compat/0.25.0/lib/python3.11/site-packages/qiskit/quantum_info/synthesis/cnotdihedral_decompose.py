# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the CNOTDihedral class.
"""

from __future__ import annotations
from qiskit.synthesis.cnotdihedral import (
    synth_cnotdihedral_two_qubits,
    synth_cnotdihedral_general,
)
from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    additional_msg="Instead, use the function qiskit.synthesis.synth_cnotdihedral_full.",
    since="0.23.0",
)
def decompose_cnotdihedral(elem):
    """DEPRECATED: Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.

    References:
        1. Shelly Garion and Andrew W. Cross, *Synthesis of CNOT-Dihedral circuits
           with optimal number of two qubit gates*, `Quantum 4(369), 2020
           <https://quantum-journal.org/papers/q-2020-12-07-369/>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    num_qubits = elem.num_qubits
    if num_qubits < 3:
        return synth_cnotdihedral_two_qubits(elem)

    return synth_cnotdihedral_general(elem)
