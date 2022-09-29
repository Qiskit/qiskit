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
Fake Almaden device (20 qubit).
"""

import os
from qiskit.providers.fake_provider import fake_qasm_backend, fake_backend


class FakeAlmadenV2(fake_backend.FakeBackendV2):
    """A fake Almaden V2 backend.

    .. code-block:: text

        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
              ↕         ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕         ↕         ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
              ↕         ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_almaden.json"
    props_filename = "props_almaden.json"
    backend_name = "fake_almaden_v2"


class FakeAlmaden(fake_qasm_backend.FakeQasmBackend):
    """A fake Almaden backend.

    .. code-block:: text

        00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
              ↕         ↕
        05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
         ↕         ↕         ↕
        10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
              ↕         ↕
        15 ↔ 16 ↔ 17 ↔ 18 ↔ 19
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_almaden.json"
    props_filename = "props_almaden.json"
    backend_name = "fake_almaden"
