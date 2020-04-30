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
Fake Boeblingen device (20 qubit).
"""

import os
import json

from qiskit.providers.models import (PulseBackendConfiguration,
                                     BackendProperties, PulseDefaults)
from qiskit.test.mock.fake_backend import FakeBackend


class FakeBoeblingen(FakeBackend):
    """A fake Boeblingen backend."""

    def __init__(self):
        """
          00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                ↕         ↕
          05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
           ↕         ↕         ↕
          10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
                ↕         ↕
          15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
        """
        dirname = os.path.dirname(__file__)
        filename = "conf_boeblingen.json"
        with open(os.path.join(dirname, filename), "r") as f_conf:
            conf = json.load(f_conf)
        configuration = PulseBackendConfiguration.from_dict(conf)
        configuration.backend_name = 'fake_boeblingen'
        self._defaults = None
        self._properties = None
        super().__init__(configuration)

    def properties(self):
        """Returns a snapshot of device properties"""
        dirname = os.path.dirname(__file__)
        filename = "props_boeblingen.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)

    def defaults(self):
        """Returns a snapshot of device defaults"""
        if not self._defaults:
            dirname = os.path.dirname(__file__)
            filename = "defs_boeblingen.json"
            with open(os.path.join(dirname, filename), "r") as f_defs:
                defs = json.load(f_defs)
            self._defaults = PulseDefaults.from_dict(defs)
        return self._defaults
