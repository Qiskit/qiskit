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
Fake Cambridge device (20 qubit).
"""

import os
import json

from qiskit.providers.models import QasmBackendConfiguration, BackendProperties
from qiskit.test.mock.fake_backend import FakeBackend


class FakeCambridge(FakeBackend):
    """A fake Cambridge backend."""

    def __init__(self):
        """
                   00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                   ↕                    ↕
                   05                  06
                   ↕                    ↕
         07 ↔ 08 ↔ 09 ↔ 10 ↔ 11 ↔ 12 ↔ 13 ↔ 14 ↔ 15
         ↕                   ↕                    ↕
         16                  17                  18
         ↕                   ↕                    ↕
         19 ↔ 20 ↔ 21 ↔ 22 ↔ 23 ↔ 24 ↔ 25 ↔ 26 ↔ 27
        """
        dirname = os.path.dirname(__file__)
        filename = "conf_cambridge.json"
        with open(os.path.join(dirname, filename), "r") as f_conf:
            conf = json.load(f_conf)

        configuration = QasmBackendConfiguration.from_dict(conf)
        configuration.backend_name = 'fake_cambridge'
        self._defaults = None
        self._properties = None
        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties"""
        if not self._properties:
            dirname = os.path.dirname(__file__)
            filename = "props_cambridge.json"
            with open(os.path.join(dirname, filename), "r") as f_prop:
                props = json.load(f_prop)
            self._properties = BackendProperties.from_dict(props)
        return self._properties
