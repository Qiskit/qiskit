# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import numpy as np

from qiskit.providers.models import PulseDefaults
from qiskit.compiler import schedule
from qiskit.circuit import QuantumCircuit, Gate


def gen_source(num_random_gate):

    # minimum data to instantiate pulse defaults.
    # cmd def contains cx, rz, sx, u3, measure
    # and random gate (custom waveform of 100 dt + 2 fc)

    qobj_dict = {
        "qubit_freq_est": [5.0, 5.1],
        "meas_freq_est": [7.0, 7.0],
        "buffer": 0,
        "pulse_library": [],
        "cmd_def": [
            {
                "name": "cx",
                "qubits": [0, 1],
                "sequence": [
                    {
                        "ch": "d0",
                        "name": "fc",
                        "phase": -3.141592653589793,
                        "t0": 0,
                    },
                    {
                        "ch": "d0",
                        "label": "Y90p_d0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.0022743565483134 + 0.14767107967944j),
                            "beta": 0.5218372954777448,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 0,
                    },
                    {
                        "ch": "d0",
                        "label": "CR90p_d0_u1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.03583301328943 - 0.0006486874906466j),
                            "duration": 1104,
                            "sigma": 64,
                            "width": 848,
                        },
                        "pulse_shape": "gaussian_square",
                        "t0": 96,
                    },
                    {
                        "ch": "d0",
                        "label": "CR90m_d0_u1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (-0.03583301328943 + 0.000648687490646j),
                            "duration": 1104,
                            "sigma": 64,
                            "width": 848,
                        },
                        "pulse_shape": "gaussian_square",
                        "t0": 1296,
                    },
                    {
                        "ch": "d0",
                        "name": "fc",
                        "phase": -1.5707963267948966,
                        "t0": 2400,
                    },
                    {
                        "ch": "d0",
                        "label": "X90p_d0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.14766707017470 - 0.002521280908868j),
                            "beta": 0.5218372954777448,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 2400,
                    },
                    {
                        "ch": "d1",
                        "name": "fc",
                        "phase": -1.5707963267948966,
                        "t0": 0,
                    },
                    {
                        "ch": "d1",
                        "label": "X90p_d1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.19074973504459 + 0.004525711677119j),
                            "beta": -1.2815198779814807,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 0,
                    },
                    {
                        "ch": "d1",
                        "label": "Xp_d1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.3872223088586379 + 0j),
                            "beta": -1.498502772395478,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 1200,
                    },
                    {
                        "ch": "d1",
                        "label": "Y90m_d1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.00285052543950 - 0.19078212177897j),
                            "beta": -1.2815198779814807,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 2400,
                    },
                    {
                        "ch": "u0",
                        "name": "fc",
                        "phase": -1.5707963267948966,
                        "t0": 0,
                    },
                    {
                        "ch": "u1",
                        "name": "fc",
                        "phase": -3.141592653589793,
                        "t0": 0,
                    },
                    {
                        "ch": "u1",
                        "label": "CR90p_u1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (-0.1629668182698 - 0.8902610676540j),
                            "duration": 1104,
                            "sigma": 64,
                            "width": 848,
                        },
                        "pulse_shape": "gaussian_square",
                        "t0": 96,
                    },
                    {
                        "ch": "u1",
                        "label": "CR90m_u1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.16296681826986 + 0.8902610676540j),
                            "duration": 1104,
                            "sigma": 64,
                            "width": 848,
                        },
                        "pulse_shape": "gaussian_square",
                        "t0": 1296,
                    },
                    {
                        "ch": "u1",
                        "name": "fc",
                        "phase": -1.5707963267948966,
                        "t0": 2400,
                    },
                ],
            },
            {
                "name": "rz",
                "qubits": [0],
                "sequence": [
                    {
                        "ch": "d0",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 0,
                    },
                    {
                        "ch": "u1",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 0,
                    },
                ],
            },
            {
                "name": "rz",
                "qubits": [1],
                "sequence": [
                    {
                        "ch": "d1",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 0,
                    },
                    {
                        "ch": "u0",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 0,
                    },
                ],
            },
            {
                "name": "sx",
                "qubits": [0],
                "sequence": [
                    {
                        "ch": "d0",
                        "label": "X90p_d0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.14766707017470 - 0.002521280908868j),
                            "beta": 0.5218372954777448,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 0,
                    }
                ],
            },
            {
                "name": "sx",
                "qubits": [1],
                "sequence": [
                    {
                        "ch": "d1",
                        "label": "X90p_d0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.19074973504459 + 0.004525711677119j),
                            "beta": -1.2815198779814807,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 0,
                    }
                ],
            },
            {
                "name": "u3",
                "qubits": [0],
                "sequence": [
                    {
                        "ch": "d0",
                        "name": "fc",
                        "phase": "-(P2)",
                        "t0": 0,
                    },
                    {
                        "ch": "d0",
                        "label": "X90p_d0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.14766707017470 - 0.002521280908868j),
                            "beta": 0.5218372954777448,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 0,
                    },
                    {
                        "ch": "d0",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 96,
                    },
                    {
                        "ch": "d0",
                        "label": "X90m_d0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (-0.14767107967944 + 0.002274356548313j),
                            "beta": 0.5218372954777448,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 96,
                    },
                    {
                        "ch": "d0",
                        "name": "fc",
                        "phase": "-(P1)",
                        "t0": 192,
                    },
                    {
                        "ch": "u1",
                        "name": "fc",
                        "phase": "-(P2)",
                        "t0": 0,
                    },
                    {
                        "ch": "u1",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 96,
                    },
                    {
                        "ch": "u1",
                        "name": "fc",
                        "phase": "-(P1)",
                        "t0": 192,
                    },
                ],
            },
            {
                "name": "u3",
                "qubits": [1],
                "sequence": [
                    {
                        "ch": "d1",
                        "name": "fc",
                        "phase": "-(P2)",
                        "t0": 0,
                    },
                    {
                        "ch": "d1",
                        "label": "X90p_d1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.19074973504459 + 0.004525711677119j),
                            "beta": -1.2815198779814807,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 0,
                    },
                    {
                        "ch": "d1",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 96,
                    },
                    {
                        "ch": "d1",
                        "label": "X90m_d1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (-0.19078212177897 - 0.002850525439509j),
                            "beta": -1.2815198779814807,
                            "duration": 96,
                            "sigma": 24,
                        },
                        "pulse_shape": "drag",
                        "t0": 96,
                    },
                    {
                        "ch": "d1",
                        "name": "fc",
                        "phase": "-(P1)",
                        "t0": 192,
                    },
                    {
                        "ch": "u0",
                        "name": "fc",
                        "phase": "-(P2)",
                        "t0": 0,
                    },
                    {
                        "ch": "u0",
                        "name": "fc",
                        "phase": "-(P0)",
                        "t0": 96,
                    },
                    {
                        "ch": "u0",
                        "name": "fc",
                        "phase": "-(P1)",
                        "t0": 192,
                    },
                ],
            },
            {
                "name": "measure",
                "qubits": [0, 1],
                "sequence": [
                    {
                        "ch": "m0",
                        "label": "M_m0",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (-0.3003200790496 + 0.3069634566518j),
                            "duration": 1792,
                            "sigma": 64,
                            "width": 1536,
                        },
                        "pulse_shape": "gaussian_square",
                        "t0": 0,
                    },
                    {
                        "ch": "m1",
                        "label": "M_m1",
                        "name": "parametric_pulse",
                        "parameters": {
                            "amp": (0.26292757124962 + 0.14446138680205j),
                            "duration": 1792,
                            "sigma": 64,
                            "width": 1536,
                        },
                        "pulse_shape": "gaussian_square",
                        "t0": 0,
                    },
                    {
                        "ch": "m0",
                        "duration": 1504,
                        "name": "delay",
                        "t0": 1792,
                    },
                    {
                        "ch": "m1",
                        "duration": 1504,
                        "name": "delay",
                        "t0": 1792,
                    },
                    {
                        "duration": 1792,
                        "memory_slot": [0, 1],
                        "name": "acquire",
                        "qubits": [0, 1],
                        "t0": 0,
                    },
                ],
            },
        ],
    }

    # add random waveform gate entries to increase overhead
    for i in range(num_random_gate):
        for qind in (0, 1):
            samples = np.random.random(100)

            gate_name = f"ramdom_gate_{i}"
            sample_name = f"random_sample_q{qind}_{i}"

            qobj_dict["pulse_library"].append(
                {
                    "name": sample_name,
                    "samples": samples,
                }
            )
            qobj_dict["cmd_def"].append(
                {
                    "name": gate_name,
                    "qubits": [qind],
                    "sequence": [
                        {
                            "ch": f"d{qind}",
                            "name": "fc",
                            "phase": "-(P0)",
                            "t0": 0,
                        },
                        {
                            "ch": f"d{qind}",
                            "label": gate_name,
                            "name": sample_name,
                            "t0": 0,
                        },
                        {
                            "ch": f"d{qind}",
                            "name": "fc",
                            "phase": "(P0)",
                            "t0": 100,
                        },
                    ],
                },
            )

    return qobj_dict


class PulseDefaultsBench:

    params = ([0, 10, 100, 1000],)
    param_names = [
        "number of random gates",
    ]

    def setup(self, num_random_gate):
        self.source = gen_source(num_random_gate)

    def time_building_defaults(self, _):
        PulseDefaults.from_dict(self.source)


class CircuitSchedulingBench:

    params = ([1, 2, 3, 15],)
    param_names = [
        "number of unit cell repetition",
    ]

    def setup(self, repeat_unit_cell):
        source = gen_source(1)
        defaults = PulseDefaults.from_dict(source)

        self.inst_map = defaults.instruction_schedule_map
        self.meas_map = [[0, 1]]
        self.dt = 0.222e-9

        rng = np.random.default_rng(123)

        qc = QuantumCircuit(2)
        for _ in range(repeat_unit_cell):
            randdom_gate = Gate("ramdom_gate_0", 1, list(rng.random(1)))
            qc.cx(0, 1)
            qc.append(randdom_gate, [0])
            qc.sx(0)
            qc.rz(1.57, 0)
            qc.append(randdom_gate, [1])
            qc.sx(1)
            qc.rz(1.57, 1)
        qc.measure_all()
        self.qc = qc

    def time_scheduling_circuits(self, _):
        schedule(
            self.qc,
            inst_map=self.inst_map,
            meas_map=self.meas_map,
            dt=self.dt,
        )
