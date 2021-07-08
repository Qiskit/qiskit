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
Fake Manhattan device (65 qubit).
"""

import os
from qiskit.test.mock import fake_pulse_backend


class FakeManhattan(fake_pulse_backend.FakePulseBackend):
    """A fake Manhattan backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_manhattan.json"
    props_filename = "props_manhattan.json"
    defs_filename = "defs_manhattan.json"
    backend_name = "fake_manhattan"


class FakeLegacyManhattan(fake_pulse_backend.FakePulseLegacyBackend):
    """A fake Manhattan backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_manhattan.json"
    props_filename = "props_manhattan.json"
    defs_filename = "defs_manhattan.json"
    backend_name = "fake_manhattan"
