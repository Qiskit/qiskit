# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fake backend abstract class for mock backends supporting OpenPulse.
"""

from abc import ABC

from qiskit.providers.models import (PulseBackendConfiguration, PulseDefaults)
from qiskit.test.mock.fake_qasm_backend import FakeQasmBackend


class FakePulseBackend(FakeQasmBackend, ABC):
    """A fake pulse backend."""

    def defaults(self):
        """Returns a snapshot of device defaults"""
        if not self._defaults:
            self._set_defaults_from_json()
        return self._defaults

    def _set_defaults_from_json(self):
        defs = self._load_json(self.defs_filename)
        self._defaults = PulseDefaults.from_dict(defs)

    def _get_config_from_dict(self, conf):
        return PulseBackendConfiguration.from_dict(conf)
