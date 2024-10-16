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
from qiskit.circuit.quantumcircuit import QuantumCircuit, _copy_metadata
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import CXGate, ECRGate, CZGate, iSwapGate
from qiskit.exceptions import QiskitError


NAME_TO_CLASS = {
    "cx": CXGate._standard_gate,
    "ecr": ECRGate._standard_gate,
    "cz": CZGate._standard_gate,
    "iswap": iSwapGate._standard_gate,
}


def twirl_circuit(
    circuit: QuantumCircuit,
    twirling_gate: None | str | type[Gate] | list[str] | list[type[Gate]] = None,
    seed: int | None = None,
    num_twirls: int | None = None,
) -> QuantumCircuit | list[QuantumCircuit]:
    """Create a copy of a given circuit with Pauli twirling applied around a specified two qubit
    gate.

    Args:
        circuit: The circuit to twirl
        twirling_gate: The gate to twirl, defaults to `None` which means twirl all supported gates.
            If supplied it can either be a single gate or a list of gates either as a gate class
            or it's string name. Currently only :class:`.CXGate` (`"cx"`), :class:`.CZGate` (`"cz"`),
            :class:`.ECRGate` (`"ecr"`), and :class:`.iSwapGate` (`"iswap"`) are supported.
        seed: An integer seed for the random number generator used internally.
            If specified this must be between 0 and 18,446,744,073,709,551,615.
        num_twirls: The number of twirling circuits to build. This defaults to None and will return
            a single circuit. If it is an integer a list of circuits with `num_twirls` circuits
            will be returned.

    Returns:
        A copy of the given circuit with Pauli twirling applied to each
        instance of the specified twirling gate.
    """
    if isinstance(twirling_gate, str):
        gate = NAME_TO_CLASS.get(twirling_gate, None)
        if gate is None:
            raise QiskitError(f"The specified gate name {twirling_gate} is not supported")
        twirling_std_gate = [gate]
    elif isinstance(twirling_gate, list):
        twirling_std_gate = []
        for gate in twirling_gate:
            if isinstance(gate, str):
                gate = NAME_TO_CLASS.get(gate, None)
                if gate is None:
                    raise QiskitError(f"The specified gate name {twirling_gate} is not supported")
                twirling_std_gate.append(gate)
            else:
                twirling_gate = getattr(gate, "_standard_gate", None)
                if twirling_gate is None:
                    raise QiskitError("This function can only be used with standard gates")
                twirling_std_gate.append(twirling_gate)
    elif twirling_gate is not None:
        twirling_std_gate = getattr(twirling_gate, "_standard_gate", None)
        if twirling_std_gate is None:
            raise QiskitError("This function can only be used with standard gates")
        twirling_std_gate = [twirling_std_gate]
    else:
        twirling_std_gate = twirling_gate
    out_twirls = num_twirls
    if out_twirls is None:
        out_twirls = 1
    new_data = twirl_rs(circuit._data, twirling_std_gate, seed, out_twirls)
    if num_twirls is not None:
        out_list = []
        for circ in new_data:
            new_circ = QuantumCircuit._from_circuit_data(circ)
            _copy_metadata(circuit, new_circ, "alike")
            out_list.append(new_circ)
        return out_list
    else:
        out_circ = QuantumCircuit._from_circuit_data(new_data[0])
        _copy_metadata(circuit, out_circ, "alike")
        return out_circ
