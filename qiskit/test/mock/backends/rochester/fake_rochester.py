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
Fake Rochester device (53 qubit).
"""

import os
import json

from qiskit.providers.models import (GateConfig, QasmBackendConfiguration,
                                     BackendProperties)
from qiskit.test.mock.fake_backend import FakeBackend


class FakeRochester(FakeBackend):
    """A fake Rochester backend."""

    def __init__(self):
        cmap = [[0, 5], [0, 1], [1, 2], [1, 0], [2, 3], [2, 1], [3, 4], [3, 2],
                [4, 6], [4, 3], [5, 9], [5, 0], [6, 13], [6, 4], [7, 16], [7, 8],
                [8, 9], [8, 7], [9, 10], [9, 8], [9, 5], [10, 11], [10, 9], [11, 17],
                [11, 12], [11, 10], [12, 13], [12, 11], [13, 14], [13, 12], [13, 6], [14, 15],
                [14, 13], [15, 18], [15, 14], [16, 19], [16, 7], [17, 23], [17, 11], [18, 27],
                [18, 15], [19, 20], [19, 16], [20, 21], [20, 19], [21, 28], [21, 22], [21, 20],
                [22, 23], [22, 21], [23, 24], [23, 22], [23, 17], [24, 25], [24, 23], [25, 29],
                [25, 26], [25, 24], [26, 27], [26, 25], [27, 26], [27, 18], [28, 32], [28, 21],
                [29, 36], [29, 25], [30, 39], [30, 31], [31, 32], [31, 30], [32, 33], [32, 31],
                [32, 28], [33, 34], [33, 32], [34, 40], [34, 35], [34, 33], [35, 36], [35, 34],
                [36, 37], [36, 35], [36, 29], [37, 38], [37, 36], [38, 41], [38, 37], [39, 42],
                [39, 30], [40, 46], [40, 34], [41, 50], [41, 38], [42, 43], [42, 39], [43, 44],
                [43, 42], [44, 51], [44, 45], [44, 43], [45, 46], [45, 44], [46, 47], [46, 45],
                [46, 40], [47, 48], [47, 46], [48, 52], [48, 49], [48, 47], [49, 50], [49, 48],
                [50, 49], [50, 41], [51, 44], [52, 48]]

        configuration = QasmBackendConfiguration(
            backend_name='fake_rochester',
            backend_version='0.0.0',
            n_qubits=53,
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
        """Returns a snapshot of device properties as recorded on 11/21/19.
        """
        dirname = os.path.dirname(__file__)
        filename = "props_rochester.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)
