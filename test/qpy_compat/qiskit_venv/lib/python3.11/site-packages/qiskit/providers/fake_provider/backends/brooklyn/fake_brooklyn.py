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
Fake Brooklyn device (65 qubit).
"""

import os
from qiskit.providers.fake_provider import fake_pulse_backend, fake_backend


class FakeBrooklynV2(fake_backend.FakeBackendV2):
    """A fake Brooklyn V2 backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_brooklyn.json"
    props_filename = "props_brooklyn.json"
    defs_filename = "defs_brooklyn.json"
    backend_name = "fake_brooklyn"


class FakeBrooklyn(fake_pulse_backend.FakePulseBackend):
    """A fake Brooklyn backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_brooklyn.json"
    props_filename = "props_brooklyn.json"
    defs_filename = "defs_brooklyn.json"
    backend_name = "fake_brooklyn"
