# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Fake Hanoi device (27 qubit).
"""

import os
from qiskit.providers.fake_provider import fake_pulse_backend, fake_backend


class FakeHanoiV2(fake_backend.FakeBackendV2):
    """A fake 27 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_hanoi.json"
    props_filename = "props_hanoi.json"
    defs_filename = "defs_hanoi.json"
    backend_name = "fake_hanoi"


class FakeHanoi(fake_pulse_backend.FakePulseBackend):
    """A fake 27 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_hanoi.json"
    props_filename = "props_hanoi.json"
    defs_filename = "defs_hanoi.json"
    backend_name = "fake_hanoi"
