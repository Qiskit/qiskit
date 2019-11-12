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
Fake Poughkeepsie device (20 qubit).
"""

import os
import json

from qiskit.providers.models import (GateConfig, QasmBackendConfiguration,
                                     BackendProperties)
from qiskit.test.mock.fake_backend import FakeBackend


class FakePoughkeepsie(FakeBackend):
    """A fake Poughkeepsie backend."""

    def __init__(self):
        """
          00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
           ↕                   ↕
          05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
           ↕         ↕         ↕
          10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
           ↕                   ↕
          15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
        """
        cmap = [[0, 1], [0, 5], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 9],
                [5, 0], [5, 6], [5, 10], [6, 5], [6, 7], [7, 6], [7, 8], [7, 12], [8, 7], [8, 9],
                [9, 4], [9, 8], [9, 14], [10, 5], [10, 11], [10, 15], [11, 10], [11, 12], [12, 7],
                [12, 11], [12, 13], [13, 12], [13, 14], [14, 9], [14, 13], [14, 19], [15, 10],
                [15, 16], [16, 15], [16, 17], [17, 16], [17, 18], [18, 17], [18, 19], [19, 14],
                [19, 18]]

        configuration = QasmBackendConfiguration(
            backend_name='fake_poughkeepsie',
            backend_version='0.0.0',
            n_qubits=20,
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=True,
            max_shots=8192,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=cmap,
        )

        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties as recorded on 10/08/19.
        """
        dirname = os.path.dirname(__file__)
        filename = "props_poughkeepsie.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)
