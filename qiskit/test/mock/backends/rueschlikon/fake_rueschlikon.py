# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fake Reuschlikon device (16 qubit).
"""

from qiskit.providers.models import GateConfig, QasmBackendConfiguration
from qiskit.test.mock.fake_backend import FakeBackend, FakeLegacyBackend


class FakeRueschlikon(FakeBackend):
    """A fake 16 qubit backend."""

    def __init__(self):
        """
        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9
        """
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [3, 4],
            [3, 14],
            [5, 4],
            [6, 5],
            [6, 7],
            [6, 11],
            [7, 10],
            [8, 7],
            [9, 8],
            [9, 10],
            [11, 10],
            [12, 5],
            [12, 11],
            [12, 13],
            [13, 4],
            [13, 14],
            [15, 0],
            [15, 2],
            [15, 14],
        ]

        configuration = QasmBackendConfiguration(
            backend_name="fake_rueschlikon",
            backend_version="0.0.0",
            n_qubits=16,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            max_experiments=900,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=cmap,
        )

        super().__init__(configuration)


class FakeLegacyRueschlikon(FakeLegacyBackend):
    """A fake 16 qubit backend."""

    def __init__(self):
        """
        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9
        """
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [3, 4],
            [3, 14],
            [5, 4],
            [6, 5],
            [6, 7],
            [6, 11],
            [7, 10],
            [8, 7],
            [9, 8],
            [9, 10],
            [11, 10],
            [12, 5],
            [12, 11],
            [12, 13],
            [13, 4],
            [13, 14],
            [15, 0],
            [15, 2],
            [15, 14],
        ]

        configuration = QasmBackendConfiguration(
            backend_name="fake_rueschlikon",
            backend_version="0.0.0",
            n_qubits=16,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            max_experiments=900,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=cmap,
        )

        super().__init__(configuration)
