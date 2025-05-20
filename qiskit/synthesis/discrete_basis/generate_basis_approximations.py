# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to generate the basic approximations of single qubit gates for Solovay-Kitaev."""

from __future__ import annotations

import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate.synthesis.discrete_basis import GateSequence

_1q_gates = {
    "i": gates.IGate(),
    "x": gates.XGate(),
    "y": gates.YGate(),
    "z": gates.ZGate(),
    "h": gates.HGate(),
    "t": gates.TGate(),
    "tdg": gates.TdgGate(),
    "s": gates.SGate(),
    "sdg": gates.SdgGate(),
    "sx": gates.SXGate(),
    "sxdg": gates.SXdgGate(),
}


@deprecate_func(
    since="2.1",
    additional_msg=(
        "Use the SolovayKitaevDecomposition class directly, to generate, store, and load the "
        "basic approximations."
    ),
    pending=True,
)
def generate_basic_approximations(
    basis_gates: list[str | Gate], depth: int, filename: str | None = None
) -> list[GateSequence]:
    """Generates a list of :class:`GateSequence`\\ s with the gates in ``basis_gates``.

    Args:
        basis_gates: The gates from which to create the sequences of gates.
        depth: The maximum depth of the approximations.
        filename: If provided, the basic approximations are stored in this file.

    Returns:
        List of :class:`GateSequence`\\ s using the gates in ``basis_gates``.

    Raises:
        ValueError: If ``basis_gates`` contains an invalid gate identifier.
    """
    from .solovay_kitaev import SolovayKitaevDecomposition

    basis = []
    for gate in basis_gates:
        if isinstance(gate, str):
            if gate not in _1q_gates:
                raise ValueError(f"Invalid gate identifier: {gate}")
            basis.append(gate)
        else:  # gate is a qiskit.circuit.Gate
            basis.append(gate.name)

    sk = SolovayKitaevDecomposition(basis_gates=basis, depth=depth)

    if filename is not None:
        sk.save_basic_approximations(filename)

    return sk._sk.get_gate_sequences()
