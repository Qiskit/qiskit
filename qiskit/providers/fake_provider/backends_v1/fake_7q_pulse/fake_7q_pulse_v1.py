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
A 7 qubit fake :class:`.BackendV1` with pulse capabilities.
"""

import os
from qiskit.providers.fake_provider import fake_pulse_backend


class Fake7QPulseV1(fake_pulse_backend.FakePulseBackend):
    """A fake **pulse** backend with the following characteristics:

    * num_qubits: 7
    * coupling_map:

        .. code-block:: text

            0 ↔ 1 ↔ 3 ↔ 5 ↔ 6
                ↕       ↕
                2       4

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'u3', 'id', 'measure', 'u2', 'x', 'u1', 'sx', 'rz'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, 1, 2, 3, 4, 5, 6)
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_nairobi.json"
    props_filename = "props_nairobi.json"
    defs_filename = "defs_nairobi.json"
    backend_name = "fake_7q_pulse_v1"
