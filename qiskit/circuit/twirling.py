# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The twirling module."""

from __future__ import annotations

from qiskit._accelerate.twirling import twirl_circuit as twirl_rs
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.exceptions import QiskitError


def twirl_circuit(
    circuit: QuantumCircuit, twirling_gate: Gate = CXGate, seed: int = None, num_twirls: int = 1,
) -> QuantumCircuit | list[QuantumCircuit]:
    """Create a copy of a given circuit with Pauli twirling applied around a specified two qubit
    gate.

    Args:
        circuit: The circuit to twirl
        twirling_gate: The gate to twirl, currently only :class:`.CXGate`, :class:`.CZGate`,
            :class:`.ECRGate`, and :class:`.iSwapGate` are supported.
        seed: An integer seed for the random number generator used internally.
            If specified this must be between 0 and 8,446,744,073,709,551,615.
        num_twirls: The number of twirling circuits to build. This defaults to 1 and will return
            a single circuit. If it is > 1 a list of circuits will be returned.

    Returns:
        A copy of the given circuit with Pauli twirling applied to each
        instance of the specified twirling gate.
    """
    twirling_std_gate = getattr(twirling_gate, "_standard_gate", None)
    if twirling_std_gate is None:
        raise QiskitError("This function can only be used with standard gates")
    new_data = twirl_rs(circuit._data, twirling_std_gate, seed, num_twirls)
    if num_twirls > 1:
        return [QuantumCircuit._from_circuit_data(x) for x in new_data]
    else:
        return QuantumCircuit._from_circuit_data(new_data[0])
