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

from qiskit.providers.models import (GateConfig, PulseBackendConfiguration,
                                     PulseDefaults, Command, UchannelLO)
from qiskit.providers.models.backendproperties import Nduv, Gate, BackendProperties
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction
from .fake_backend import FakeBackend


class FakeOpenPulse2Q(FakeBackend):
    """A fake 2 qubit backend for pulse test."""

    def __init__(self):
        configuration = PulseBackendConfiguration(
            backend_name='fake_openpulse_2q',
            backend_version='0.0.0',
            n_qubits=2,
            meas_levels=[0, 1, 2],
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=True,
            open_pulse=True,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map=[[0, 1]],
            n_registers=2,
            n_uchannels=2,
            u_channel_lo=[
                [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)]
            ],
            meas_level=[1, 2],
            qubit_lo_range=[[4.5, 5.5], [4.5, 5.5]],
            meas_lo_range=[[6.0, 7.0], [6.0, 7.0]],
            dt=1.3333,
            dtm=10.5,
            rep_times=[100, 250, 500, 1000],
            meas_map=[[0, 1]],
            channel_bandwidth=[
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3],
                [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]
            ],
            meas_kernels=['kernel1'],
            discriminators=['max_1Q_fidelity'],
            acquisition_latency=[[100, 100], [100, 100]],
            conditional_latency=[
                [100, 1000], [1000, 100], [100, 1000],
                [1000, 100], [100, 1000], [1000, 100]
            ],
            hamiltonian={
                'h_str': ["np.pi*(2*v0-alpha0)*O0", "np.pi*alpha0*O0*O0", "2*np.pi*r*X0||D0",
                          "2*np.pi*r*X0||U1", "2*np.pi*r*X1||U0", "np.pi*(2*v1-alpha1)*O1",
                          "np.pi*alpha1*O1*O1", "2*np.pi*r*X1||D1", "2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)"],
                'description': "A hamiltonian for a mocked 2Q device, with 1Q and 2Q terms.",
                'qub': {'0': 3, '1': 3},
                'vars':  {'v0': 5.00,
                          'v1': 5.1,
                          'j': 0.01,
                          'r': 0.02,
                          'alpha0': -0.33,
                          'alpha1': -0.33}
            }
        )

        self._defaults = PulseDefaults(
            qubit_freq_est=[4.9, 5.0],
            meas_freq_est=[6.5, 6.6],
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
        dt = 1.3333  # pylint: disable=invalid-name
        self._properties = BackendProperties(
            backend_name='fake_openpulse_2q',
            backend_version='0.0.0',
            last_update_date=mock_time,
            qubits=[
                [Nduv(date=mock_time, name='T1', unit='µs', value=71.9500421005539),
                 Nduv(date=mock_time, name='frequency', unit='MHz', value=4919.96800692)],
                [Nduv(date=mock_time, name='T1', unit='µs', value=81.9500421005539),
                 Nduv(date=mock_time, name='frequency', unit='GHz', value=5.01996800692)]
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
            ],
            general=[]
        )

        super().__init__(configuration)

    def defaults(self):
        """Return the default pulse-related settings provided by the backend (such as gate
        to Schedule mappings).
        """
        return self._defaults

    def properties(self):
        """Return the measured characteristics of the backend."""
        return self._properties
