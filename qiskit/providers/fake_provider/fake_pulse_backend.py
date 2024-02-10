# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
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

import warnings

from qiskit.exceptions import QiskitError
from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults

from .fake_qasm_backend import FakeQasmBackend
from .utils.json_decoder import decode_pulse_defaults


class FakePulseBackend(FakeQasmBackend):
    """A fake pulse backend."""

    defs_filename = None

    def __init__(self):
        super().__init__()
        # This is a deprecation warning for the subclasses.
        # FakePulseBackend is not deprecated.
        warnings.warn(
            message="All fake backend instances based on real device snapshots (`FakeVigo`,"
            "`FakeSherbrooke`,...) have been migrated to the `qiskit_ibm_runtime` package. "
            "These classes are deprecated as of qiskit 0.46.0 and will be removed in qiskit 1.0.0. "
            "To migrate your code, run `pip install qiskit-ibm-runtime` and use "
            "`from qiskit_ibm_runtime.fake_provider import FakeExample` "
            "instead of `from qiskit.providers.fake_provider import FakeExample`. "
            "If you are using a custom fake backend implementation, you don't need to take any action.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    def defaults(self):
        """Returns a snapshot of device defaults"""
        if not self._defaults:
            self._set_defaults_from_json()
        return self._defaults

    def _set_defaults_from_json(self):
        if not self.props_filename:
            raise QiskitError("No properties file has been defined")
        defs = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs)
        self._defaults = PulseDefaults.from_dict(defs)

    def _get_config_from_dict(self, conf):
        return PulseBackendConfiguration.from_dict(conf)
