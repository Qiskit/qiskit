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
Fake Sydney device (27 qubit).
"""

import os
from qiskit.test.mock import fake_pulse_backend


class FakeSydney(fake_pulse_backend.FakePulseBackend):
    """A fake 27 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_sydney.json"
    props_filename = "props_sydney.json"
    defs_filename = "defs_sydney.json"
    backend_name = "fake_sydney"


class FakeLegacySydney(fake_pulse_backend.FakePulseLegacyBackend):
    """A fake 27 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_sydney.json"
    props_filename = "props_sydney.json"
    defs_filename = "defs_sydney.json"
    backend_name = "fake_sydney"
