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
Fake Paris device (20 qubit).
"""

import os
from qiskit.test.mock import fake_pulse_backend


class FakeParis(fake_pulse_backend.FakePulseBackend):
    """A fake Paris backend.

                   06                  17
                   ↕                    ↕
    00 ↔ 01 ↔ 04 ↔ 07 ↔ 10 ↔ 12 ↔ 15 ↔ 18 ↔ 20 ↔ 23
         ↕                   ↕                    ↕
         02                  13                  24
         ↕                   ↕                    ↕
         03 ↔ 05 ↔ 08 ↔ 11 ↔ 14 ↔ 16 ↔ 19 ↔ 22 ↔ 25 ↔ 26
                   ↕                    ↕
                   09                  20
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_paris.json"
    props_filename = "props_paris.json"
    defs_filename = "defs_paris.json"
    backend_name = "fake_paris"


class FakeLegacyParis(fake_pulse_backend.FakePulseLegacyBackend):
    """A fake Paris backend.

                   06                  17
                   ↕                    ↕
    00 ↔ 01 ↔ 04 ↔ 07 ↔ 10 ↔ 12 ↔ 15 ↔ 18 ↔ 20 ↔ 23
         ↕                   ↕                    ↕
         02                  13                  24
         ↕                   ↕                    ↕
         03 ↔ 05 ↔ 08 ↔ 11 ↔ 14 ↔ 16 ↔ 19 ↔ 22 ↔ 25 ↔ 26
                   ↕                    ↕
                   09                  20
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_paris.json"
    props_filename = "props_paris.json"
    defs_filename = "defs_paris.json"
    backend_name = "fake_paris"
