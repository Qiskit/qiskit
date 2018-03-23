# pylint: disable=invalid-name,missing-docstring,broad-except

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Quantum Program QISKit Test."""

import os
import unittest
import qiskit.backends
from qiskit import (ClassicalRegister, QISKitError, QuantumCircuit,
                    QuantumRegister, QuantumProgram, Result, QuantumJob,
                    RegisterSizeError)
from .common import requires_qe_access, QiskitTestCase, Path
import jsonschema

class TestQuantumJob(QiskitTestCase):
    """Test QuantumJob class"""

    def setUp(self):
        qr = QuantumRegister('qr', 2)
        cr = ClassicalRegister('cr', 2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)
        self.qc = qc

    def test_valid_qobj_backend_specific(self):
        p_reset = 0
        p_m0 = 0.5
        p_m1 = 0.5
        r = 0.1
        p0 = 0.9
        p1 = 0.1
        gate_error = {"p_depol": 0.1,
                      "p_pauli": [0.1, 0.1, 0.1, 0.1],
                      "gate_time": 1,
                      "U_error": [[1,0], [0,1]]
        }
        x90_gate_error = gate_error.copy()
        x90_gate_error.update({"calibration_error": 0.1,
                               "detuning_error": 0.1})
        config = {
            "shots": 4,
            "seed": 0,
            "shots_threads": 4,
            "data": [
                "classical_states",
                "quantum_states",
                "density_matrix",
                "overlaps",
                "probabilities",
                "probabilities_ket"
            ],
            "initial_state": [1, 0, 0, 1],
            "target_states": [
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [[1, 0], [0, 0], [0, 0], [0, 1]],
                [[1, 0], [0, 0], [0, 0], [0, -1]]
            ],
            "renorm_target_states": True,
            "chop": 1e-10,
            "noise_params": {
                "reset_error": p_reset,
                "readout_error": [p_m0, p_m1],
                "relaxation_rate": r,
                "thermal_populations": [p0, p1],
                "measure": gate_error,
                "reset": gate_error,
                "id": gate_error,
                "U": gate_error,
                "X90": x90_gate_error,
                "CX": {
                    "p_depol": 0.1,
                    "p_pauli": [0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1],
                    "gate_time": 1,
                    "U_error": [[1,0], [0,1]],
                    #"amp_error": 0.1,
                    "zz_error": 0.1
                }
            }
        }
        q_job = QuantumJob(self.qc, config=config,
                           backend='local_qiskit_simulator')

    def test_invalid_qobj_backend_specific(self):
        p_reset = 0.5
        p_m0 = 0.5
        p_m1 = 0.5
        r = 0.1
        p0 = 0.9
        p1 = 0.1
        gate_error = {"p_depol": .1,
                      "p_pauli": [0.1, 0.1, 0.1, 0.1],
                      "gate_time": 1,
                      "U_error": [[1,0], [0,1]]
        }
        x90_gate_error = gate_error.copy()
        x90_gate_error.update({"calibration_error": 0.1,
                               "detuning_error": 0.1})
        config = {
            "shots": 4,
            "seed": 0,
            "shots_threads": 4,
            "data": [
                "classical_states",
                "quantum_states",
                "saved_quantum_states",
                "probabilities"
            ],
            "initial_state": [1, 0, 0, 1],
            "target_states": [
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [[1, 0], [0, 0], [0, 0], [0, 1]],
                [[1, 0], [0, 0], [0, 0], [0, -1]]
            ],
            "renorm_target_states": True,
            "chop": 1e-10,
            "noise_params": {
                "reset_error": p_reset,
                "readout_error": [p_m0, p_m1],
                "relaxation_rate": r,
                "thermal_populations": [p0, p1],
                "measure": gate_error,
                "reset": gate_error,
                "id": gate_error,
                "U": gate_error,
                "X90": x90_gate_error,
                "CX": {
                    "p_depol": 0.1,
                    "p_pauli": [0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1],
                    "gate_time": 1,
                    "U_error": [[1,0], [0,1]],
                    "amp_error": 0.1,
                    "zz_error": 0.1
                }
            }
        }
        # "amp_error" is not allowed for CX
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            q_job = QuantumJob(self.qc, config=config,
                               backend='local_qiskit_simulator')

    
