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
Fake Yorktown device (5 qubit).
"""

import os
from qiskit.test.mock import fake_qasm_backend


class FakeYorktown(fake_qasm_backend.FakeQasmBackend):
    """A fake 5 qubit backend.

        1
      / |
    0 - 2 - 3
        | /
        4
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_yorktown.json"
    props_filename = "props_yorktown.json"
    backend_name = "fake_yorktown"


class FakeLegacyYorktown(fake_qasm_backend.FakeQasmLegacyBackend):
    """A fake 5 qubit backend.

        1
      / |
    0 - 2 - 3
        | /
        4
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_yorktown.json"
    props_filename = "props_yorktown.json"
    backend_name = "fake_yorktown"
