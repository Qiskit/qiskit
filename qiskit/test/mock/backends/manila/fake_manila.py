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
Fake Manila device (5 qubit).
"""

import os
<<<<<<< HEAD
<<<<<<< HEAD
from qiskit.test.mock import fake_pulse_backend, fake_backend


class FakeManilaV2(fake_backend.FakeBackendV2):
=======
from qiskit.test.mock import fake_qasm_backend


class FakeManila(fake_qasm_backend.FakeQasmBackend):
>>>>>>> 0018e5f8ea5a8ff60d855ca8b317a1b1e27a83da
    """A fake 5 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_manila.json"
    props_filename = "props_manila.json"
<<<<<<< HEAD
    defs_filename = "defs_manila.json"
    backend_name = "fake_manila_v2"
=======
from qiskit.test.mock import fake_qasm_backend
>>>>>>> 8b57d7703 (Revert "Working update")


class FakeManila(fake_qasm_backend.FakeQasmBackend):
    """A fake 5 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_manila.json"
    props_filename = "props_manila.json"
=======
>>>>>>> 0018e5f8ea5a8ff60d855ca8b317a1b1e27a83da
    backend_name = "fake_manila"
