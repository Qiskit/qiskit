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

import datetime

from scipy import signal
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt 
import numpy as np
from qiskit.providers.models import (GateConfig, PulseBackendConfiguration,
                                     PulseDefaults, Command, UchannelLO)
from qiskit.providers.models.backendproperties import Nduv, Gate, BackendProperties                                    
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction 
from qiskit.pulse.commands import DelayInstruction
from qiskit.test.mock.fake_backend import FakeBackend


class FakeNVCenters2Q(FakeBackend):
    """Trivial extension of the FakeOpenPulse2Q to a fake 2 qubit NVCenter backend for pulse test."""

    def __init__(self):
        configuration = PulseBackendConfiguration(
            backend_name='fake_nvcenters_2q',
            backend_version='0.0.0',
            n_qubits=2,
            meas_levels=[0, 1, 2],
            basis_gates=['u1', 'u2', 'u3', 'cx', 'conditional_x', 'conditional_y', 'unconditional_x', 'unconditional_y', 'id'],
            simulator=False,
            local=True,
            conditional=True,
            open_pulse=True,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map= [[0, 1]],
            n_registers=2,
            n_uchannels=2,
            u_channel_lo=[ [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=1. + 0.j)]
            ],
            meas_level=[0, 1, 2],
            qubit_lo_range=[[2.5, 3.0], [2.5, 3.0]],
            meas_lo_range=[[3.2, 4.0], [3.2, 4.0]],
            dt=0.348,
            dtm=3.0,
            rep_times=[100, 250, 500, 1000],
            meas_map=[[0, 1]],
            channel_bandwidth=[
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3],
                [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02],
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3]
            ],
            meas_kernels=['default'],
            discriminators=['max_2Q_fidelity'],
            acquisition_latency=[[100, 100], [100, 100], [100, 100]],
            conditional_latency=[
                [100, 1000], [1000, 100], [100, 1000],
                [100, 1000], [1000, 100], [100, 1000],
                [1000, 100], [100, 1000], [1000, 100]
            ]
        )

        self._defaults = PulseDefaults(
            qubit_freq_est=[2.87, 3.0, 2.75],
            meas_freq_est=[3.5, 4.0, 3.2],
            buffer=10,
            pulse_library=[PulseLibraryItem(name='test_pulse_1', samples=[0.j, 0.1j]),
                           PulseLibraryItem(name='test_pulse_2', samples=[0.j, 0.1j, 1j]),
                           PulseLibraryItem(name='test_pulse_3', samples=[0.j, 0.1j, 1j, 0.5 + 0j]),
                           PulseLibraryItem(name='test_pulse_4', samples=7*[0.j, 0.1j, 1j, 0.5 + 0j]),
                           PulseLibraryItem(name='conditional_x', samples=[0.5j]*2656),
                           PulseLibraryItem(name='conditional_y', samples=[0.5j]*2656*2),
                           PulseLibraryItem(name='unconditional_x', samples=[0.5j]*3186),
                           PulseLibraryItem(name='unconditional_y', samples=[0.5j]*3186*2)],
            cmd_def=[Command(name='conditional_x', qubits=[0, 1],
                             sequence=[PulseQobjInstruction(name='conditional_x', ch='u0',t0=0, phase='0')]),
                    Command(name='conditional_y', qubits=[0, 1],
                             sequence=[PulseQobjInstruction(name='conditional_y', ch='u0',t0=0, phase='np.pi/2')]),
                    Command(name='uconditional_x',qubits=[0, 1],
                              sequence=[PulseQobjInstruction(name='unconditional_x', ch='u0',t0=0, phase='0')]),
                    Command(name='unconditional_y', qubits=[0, 1],
                              sequence=[PulseQobjInstruction(name='unconditional_y', ch='u0',t0=0, phase='np.pi/2')]),
                    Command(name='u1', qubits=[0],
                             sequence=[PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P1*np.pi')]),
                    Command(name='u1', qubits=[1],
                             sequence=[PulseQobjInstruction(name='fc', ch='d1',
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
                    Command(name='u2', qubits=[0],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d0', t0=0)]),
                    Command(name='u3', qubits=[1],
                             sequence=[PulseQobjInstruction(name='test_pulse_3', ch='d1', t0=0)]),
                    Command(name='ParametrizedGate', qubits=[0, 1],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_2', ch='u0', t0=10),
                                       PulseQobjInstruction(name='pv', ch='d1',
                                                            t0=2, val='cos(P2)'),
                                       PulseQobjInstruction(name='test_pulse_1', ch='d1', t0=20),
                                       PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=20, phase=2.1)]),
                    Command(name='measure', qubits=[0, 1],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='m0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m1', t0=0),
                                       PulseQobjInstruction(name='acquire', duration=10, t0=0,
                                                            qubits=[0, 1], memory_slot=[0, 1])])]
        )

        mock_time = datetime.datetime.now()
        dt = 0.348  # pylint: disable=invalid-name
        self._properties = BackendProperties(
            backend_name='fake_nvcenters_2q',
            backend_version='0.0.0',
            last_update_date=mock_time,
            qubits=[
                [Nduv(date=mock_time, name='T1', unit='µs', value=2530.9500421005539),
                 Nduv(date=mock_time, name='frequency', unit='MHz', value=2.84600000000)],
                [Nduv(date=mock_time, name='T1', unit='µs', value=2860.00000000000),
                 Nduv(date=mock_time, name='frequency', unit='GHz', value=0.03125000000)]
            ],
            gates=[
                Gate(gate='u1', name='u1_0', qubits=[0],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=0.)]),
                Gate(gate='u3', name='u3_0', qubits=[0],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=2 * dt)]),
                Gate(gate='u3', name='u3_1', qubits=[1],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=4 * dt)]),
                Gate(gate='cx', name='cx0_1', qubits=[0, 1],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=22 * dt)]),
                Gate(gate='conditional_x', name='conditional_x0_1', qubits=[0, 1],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=22 * dt)]),
                Gate(gate='conditional_y', name='conditional_y0_1', qubits=[0, 1],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=22 * dt)]),
                Gate(gate='unconditional_x', name='unconditional_x0_1', qubits=[0, 1],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=22 * dt)]),
                Gate(gate='unconditional_y', name='unconditional_y0_1', qubits=[0, 1],
                     parameters=[
                         Nduv(date=mock_time, name='gate_error', unit='', value=1.0),
                         Nduv(date=mock_time, name='gate_length', unit='ns', value=22 * dt)]),
            ],
            general=[]
        )
        super().__init__(configuration)


    def defaults(self):  # pylint: disable=missing-docstring
        return self._defaults
