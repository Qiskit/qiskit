# -*- coding: utf-8 -*-

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
Fake Tokyo device (20 qubit).
"""

import os
import json

from qiskit.providers.models import (GateConfig, QasmBackendConfiguration,
                                     BackendProperties)
from qiskit.test.mock.fake_backend import FakeBackend


class FakeTokyo(FakeBackend):
    """A fake 20 qubit backend."""

    def __init__(self):
        """
          00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
           ↕    ↕    ↕    ↕ ⤫  ↕
          05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
           ↕ ⤫ ↕    ↕ ⤫ ↕
          10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
           ↕    ↕ ⤫      ↕ ⤫  ↕
          15 ↔ 16 ↔ 17   18   19
        """
        cmap = [[0, 1], [0, 5], [1, 0], [1, 2], [1, 6], [2, 1],
                [2, 3], [2, 6], [3, 2], [3, 8], [3, 9], [4, 8], [4, 9],
                [5, 0], [5, 6], [5, 10], [5, 11], [6, 1], [6, 2], [6, 5],
                [6, 7], [6, 10], [6, 11], [7, 1], [7, 6], [7, 8], [7, 12],
                [7, 13], [8, 3], [8, 4], [8, 7], [8, 9], [8, 12], [8, 13],
                [9, 3], [9, 4], [9, 8], [10, 5], [10, 6], [10, 11], [10, 15],
                [11, 5], [11, 6], [11, 10], [11, 12], [11, 16], [11, 17],
                [12, 7], [12, 8], [12, 11], [12, 13], [12, 16], [13, 7],
                [13, 8], [13, 12], [13, 14], [13, 18], [13, 19], [14, 13],
                [14, 18], [14, 19], [15, 10], [15, 16], [16, 11], [16, 12],
                [16, 15], [16, 17], [17, 11], [17, 16], [18, 13], [18, 14],
                [19, 13], [19, 14]]

        configuration = QasmBackendConfiguration(
            backend_name='fake_tokyo',
            backend_version='0.0.0',
            n_qubits=20,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        """
        dirname = os.path.dirname(__file__)
        filename = "props_tokyo.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)
