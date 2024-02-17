# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A 20 qubit fake :class:`.BackendV1` without pulse capabilities.
"""

import os
from qiskit.providers.fake_provider import fake_qasm_backend


class Fake20QV1(fake_qasm_backend.FakeQasmBackend):
    """A fake backend with the following characteristics:

    * num_qubits: 20
    * coupling_map:

        .. code-block:: text

            00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                  ↕         ↕
            05 ↔ 06 ↔ 07 ↔ 08 ↔ 09
             ↕         ↕         ↕
            10 ↔ 11 ↔ 12 ↔ 13 ↔ 14
                  ↕         ↕
            15 ↔ 16 ↔ 17 ↔ 18 ↔ 19

    * basis_gates: ``["id", "u1", "u2", "u3", "cx"]``
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_singapore.json"
    props_filename = "props_singapore.json"
    backend_name = "fake_20q_v1"
