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
Fake Ourense device (5 qubit).
"""

import os
import json

from qiskit.providers.models import QasmBackendConfiguration, BackendProperties
from qiskit.test.mock.fake_backend import FakeBackend


class FakeOurense(FakeBackend):
    """A fake 5 qubit backend."""

    def __init__(self):
        """
         0 ↔ 1 ↔ 3 ↔ 4
             ↕
             2
        """
        dirname = os.path.dirname(__file__)
        filename = "conf_ourense.json"
        with open(os.path.join(dirname, filename), "r") as f_conf:
            conf = json.load(f_conf)
        configuration = QasmBackendConfiguration.from_dict(conf)
        configuration.backend_name = 'fake_ourense'
        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties"""
        dirname = os.path.dirname(__file__)
        filename = "props_ourense.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)
