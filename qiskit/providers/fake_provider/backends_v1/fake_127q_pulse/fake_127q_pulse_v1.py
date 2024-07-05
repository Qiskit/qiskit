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
A 127 qubit fake :class:`.BackendV1` with pulse capabilities.
"""

import os
from qiskit.providers.fake_provider import fake_pulse_backend


class Fake127QPulseV1(fake_pulse_backend.FakePulseBackend):
    """A fake **pulse** backend with the following characteristics:

    * num_qubits: 127
    * coupling_map: heavy-hex based
    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'id', 'measure', 'u2', 'rz', 'x', 'u3', 'sx', 'u1'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, ..., 127)
    """

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_washington.json"
    props_filename = "props_washington.json"
    defs_filename = "defs_washington.json"
    backend_name = "fake_127q_pulse_v1"
