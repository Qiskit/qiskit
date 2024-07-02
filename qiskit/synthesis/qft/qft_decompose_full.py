# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for a QFT circuit.
"""

from typing import Optional
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit


def synth_qft_full(
    num_qubits: int,
    do_swaps: bool = True,
    approximation_degree: int = 0,
    insert_barriers: bool = False,
    inverse: bool = False,
    name: Optional[str] = None,
) -> QuantumCircuit:
    """Construct a QFT circuit using all-to-all connectivity.

    Args:
        num_qubits: The number of qubits on which the QFT acts.
        do_swaps: Whether to include the final swaps in the QFT.
        approximation_degree: The degree of approximation (0 for no approximation).
        insert_barriers: If True, barriers are inserted as visualization improvement.
        inverse: If True, the inverse Fourier transform is constructed.
        name: The name of the circuit.
    """

    circuit = QuantumCircuit(num_qubits, name=name)

    for j in reversed(range(num_qubits)):
        circuit.h(j)
        num_entanglements = max(0, j - max(0, approximation_degree - (num_qubits - j - 1)))
        for k in reversed(range(j - num_entanglements, j)):
            # Use negative exponents so that the angle safely underflows to zero, rather than
            # using a temporary variable that overflows to infinity in the worst case.
            lam = np.pi * (2.0 ** (k - j))
            circuit.cp(lam, j, k)

        if insert_barriers:
            circuit.barrier()

    if do_swaps:
        for i in range(num_qubits // 2):
            circuit.swap(i, num_qubits - i - 1)

    if inverse:
        circuit = circuit.inverse()

    return circuit
