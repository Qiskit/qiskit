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
Fake Bogota device (5 qubit).
"""

import os
<<<<<<< HEAD
<<<<<<< HEAD
from qiskit.test.mock import fake_pulse_backend, fake_backend
=======
>>>>>>> 8b57d7703 (Revert "Working update")

from qiskit.test.mock import fake_qasm_backend

<<<<<<< HEAD
class FakeBogotaV2(fake_backend.FakeBackendV2):
=======

class FakeBogota(fake_qasm_backend.FakeQasmBackend):
>>>>>>> 8b57d7703 (Revert "Working update")
=======

from qiskit.test.mock import fake_qasm_backend


class FakeBogota(fake_qasm_backend.FakeQasmBackend):
>>>>>>> 0018e5f8ea5a8ff60d855ca8b317a1b1e27a83da
    """A fake 5 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_bogota.json"
    props_filename = "props_bogota.json"
<<<<<<< HEAD
<<<<<<< HEAD
    defs_filename = "defs_bogota.json"
    backend_name = "fake_bogota_v2"


class FakeBogota(fake_pulse_backend.FakePulseBackend):
=======
=======
>>>>>>> 0018e5f8ea5a8ff60d855ca8b317a1b1e27a83da
    backend_name = "fake_bogota"


class FakeLegacyBogota(fake_qasm_backend.FakeQasmLegacyBackend):
<<<<<<< HEAD
>>>>>>> 8b57d7703 (Revert "Working update")
=======
>>>>>>> 0018e5f8ea5a8ff60d855ca8b317a1b1e27a83da
    """A fake 5 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_bogota.json"
    props_filename = "props_bogota.json"
    backend_name = "fake_bogota"
