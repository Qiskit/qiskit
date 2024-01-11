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
Fake Ourense device (5 qubit).
"""

import os
from qiskit.providers.fake_provider import fake_qasm_backend, fake_backend


class FakeOurenseV2(fake_backend.FakeBackendV2):
    """A fake 5 qubit backend.

    .. code-block:: text

        0 ↔ 1 ↔ 3 ↔ 4
            ↕
            2
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_ourense.json"
    props_filename = "props_ourense.json"
    backend_name = "fake_ourense"


class FakeOurense(fake_qasm_backend.FakeQasmBackend):
    """A fake 5 qubit backend.

    .. code-block:: text

        0 ↔ 1 ↔ 3 ↔ 4
            ↕
            2
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_ourense.json"
    props_filename = "props_ourense.json"
    backend_name = "fake_ourense"
