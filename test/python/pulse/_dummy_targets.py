# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pulse target for unit tests."""

from qiskit.pulse.compiler.target import QiskitPulseTarget


TWOQ_CROSSRES_TARGET = QiskitPulseTarget(
    qubit_frames={
        0: "Q0",
        1: "Q1",
    },
    meas_frames={
        0: "M0",
        1: "M1",
    },
    tx_ports={
        "Q_channel-0": {
            "qubits": [0],
            "op_type": "generic",
            "num_frames": 3, 
            "reserved_frames": ["Q0", "Q1"],
        },
        "Q_channel-1": {
            "qubits": [1],
            "op_type": "generic",
            "num_frames": 3, 
            "reserved_frames": ["Q1"],
        },
        "R_channel-0": {
            "qubits": [0],
            "op_type": "measure",
            "num_frames": 1, 
            "reserved_frames": ["M0"],
        },
        "R_channel-1": {
            "qubits": [1],
            "op_type": "measure",
            "num_frames": 1, 
            "reserved_frames": ["M1"],
        },
    },
)
"""Pedagogical target for two qubit device control with cross resonance."""
