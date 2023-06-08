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
Fake Melbourne device (14 qubit).
"""

import os
import json

from qiskit.providers.models import GateConfig, QasmBackendConfiguration, BackendProperties
from qiskit.providers.fake_provider.fake_backend import FakeBackend
from qiskit.providers.fake_provider import fake_backend


class FakeMelbourneV2(fake_backend.FakeBackendV2):
    """A fake 14 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_melbourne.json"
    props_filename = "props_melbourne.json"
    backend_name = "fake_melbourne"


class FakeMelbourne(FakeBackend):
    """A fake 14 qubit backend."""

    def __init__(self):
        """

        .. code-block:: text

            0 ← 1 →  2 →  3 ←  4 ← 5 → 6
                ↑    ↑    ↑    ↓   ↓   ↓
               13 → 12 ← 11 → 10 ← 9 → 8 ← 7
        """
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        configuration = QasmBackendConfiguration(
            backend_name="fake_melbourne",
            backend_version="0.0.0",
            n_qubits=14,
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

    def properties(self):
        """Returns a snapshot of device properties"""
        dirname = os.path.dirname(__file__)
        filename = "props_melbourne.json"
        with open(os.path.join(dirname, filename)) as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)
