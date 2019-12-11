# -*- coding: utf-8 -*-

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
Fake backend supporting OpenPulse.
"""

from qiskit.providers.models import (GateConfig, PulseBackendConfiguration,
                                     PulseDefaults, Command, UchannelLO)
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction
from .fake_backend import FakeBackend


class FakeOpenPulse3Q(FakeBackend):
    """Trivial extension of the FakeOpenPulse2Q."""

    def __init__(self):
        configuration = PulseBackendConfiguration(
            backend_name='fake_openpulse_3q',
            backend_version='0.0.0',
            n_qubits=3,
            meas_levels=[0, 1, 2],
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=True,
            open_pulse=True,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=[[0, 1], [1, 2]],
            n_registers=3,
            n_uchannels=3,
            u_channel_lo=[
                [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=1. + 0.j)]
            ],
            meas_level=[1, 2],
            qubit_lo_range=[[4.5, 5.5], [4.5, 5.5], [4.5, 5.5]],
            meas_lo_range=[[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]],
            dt=1.3333,
            dtm=10.5,
            rep_times=[100, 250, 500, 1000],
            meas_map=[[0, 1, 2]],
            channel_bandwidth=[
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3],
                [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02],
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3]
            ],
            meas_kernels=['kernel1'],
            discriminators=['max_1Q_fidelity'],
            acquisition_latency=[[100, 100], [100, 100], [100, 100]],
            conditional_latency=[
                [100, 1000], [1000, 100], [100, 1000],
                [100, 1000], [1000, 100], [100, 1000],
                [1000, 100], [100, 1000], [1000, 100]
            ]
        )

        self._defaults = PulseDefaults(
            qubit_freq_est=[4.9, 5.0, 4.8],
            meas_freq_est=[6.5, 6.6, 6.4],
            buffer=10,
            pulse_library=[PulseLibraryItem(name='test_pulse_1', samples=[0.j, 0.1j]),
                           PulseLibraryItem(name='test_pulse_2', samples=[0.j, 0.1j, 1j]),
                           PulseLibraryItem(name='test_pulse_3',
                                            samples=[0.j, 0.1j, 1j, 0.5 + 0j]),
                           PulseLibraryItem(name='test_pulse_4',
                                            samples=7*[0.j, 0.1j, 1j, 0.5 + 0j])],
            cmd_def=[Command(name='u1', qubits=[0],
                             sequence=[PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P1*np.pi')]),
                     Command(name='u1', qubits=[1],
                             sequence=[PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=0, phase='-P1*np.pi')]),
                     Command(name='u1', qubits=[2],
                             sequence=[PulseQobjInstruction(name='fc', ch='d2',
                                                            t0=0, phase='-P1*np.pi')]),
                     Command(name='u2', qubits=[0],
                             sequence=[PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P0*np.pi'),
                                       PulseQobjInstruction(name='test_pulse_4', ch='d0', t0=0),
                                       PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P1*np.pi')]),
                     Command(name='u2', qubits=[1],
                             sequence=[PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=0, phase='-P0*np.pi'),
                                       PulseQobjInstruction(name='test_pulse_4', ch='d1', t0=0),
                                       PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=0, phase='-P0*np.pi')]),
                     Command(name='u2', qubits=[2],
                             sequence=[PulseQobjInstruction(name='test_pulse_3', ch='d2', t0=0),
                                       PulseQobjInstruction(name='fc', ch='d2',
                                                            t0=0, phase='-P0*np.pi')]),
                     Command(name='u3', qubits=[0],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d0', t0=0)]),
                     Command(name='u3', qubits=[1],
                             sequence=[PulseQobjInstruction(name='test_pulse_3', ch='d1', t0=0)]),
                     Command(name='cx', qubits=[0, 1],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_2', ch='u0', t0=10),
                                       PulseQobjInstruction(name='test_pulse_1', ch='d1', t0=20),
                                       PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=20, phase=2.1)]),
                     Command(name='cx', qubits=[1, 2],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d1', t0=0),
                                       PulseQobjInstruction(name='test_pulse_2', ch='u1', t0=10),
                                       PulseQobjInstruction(name='test_pulse_1', ch='d2', t0=20),
                                       PulseQobjInstruction(name='fc', ch='d2',
                                                            t0=20, phase=2.1)]),
                     Command(name='ParametrizedGate', qubits=[0, 1],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_2', ch='u0', t0=10),
                                       PulseQobjInstruction(name='pv', ch='d1',
                                                            t0=2, val='cos(P2)'),
                                       PulseQobjInstruction(name='test_pulse_1', ch='d1', t0=20),
                                       PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=20, phase=2.1)]),
                     Command(name='measure', qubits=[0, 1, 2],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='m0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m1', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m2', t0=0),
                                       PulseQobjInstruction(name='acquire', duration=10, t0=0,
                                                            qubits=[0, 1, 2],
                                                            memory_slot=[0, 1, 2])])]
        )

        super().__init__(configuration)

    def defaults(self):  # pylint: disable=missing-docstring
        return self._defaults
