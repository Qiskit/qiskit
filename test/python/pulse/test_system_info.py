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
"""Test that the SystemInfo methods work as expected with a mocked Pulse backend."""
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeProvider
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj import PulseQobjInstruction
from qiskit.pulse import SamplePulse, Schedule, PulseError
from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.schedule import ParameterizedSchedule

from qiskit.pulse.system_info import SystemInfo


class TestSystemInfo(QiskitTestCase):
    """Test the SystemInfo class."""

    def setUp(self):
        self.provider = FakeProvider()
        self.backend = self.provider.get_backend('fake_openpulse_2q')
        self.sysinfo = SystemInfo(self.backend)

    def test_init(self):
        """Test `init` with default_ops."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.sysinfo.drives(0)))
        sysinfo = SystemInfo(default_ops={('tmp', 0): sched})
        self.assertTrue(sysinfo.has('tmp', 0))

    def test_simple_properties(self):
        """Test the most basic getters."""
        self.assertEqual(self.sysinfo.name, self.backend.properties().backend_name)
        self.assertEqual(self.sysinfo.version, self.backend.properties().backend_version)
        self.assertEqual(self.sysinfo.n_qubits, self.backend.configuration().n_qubits)
        self.assertEqual(self.sysinfo.dt, self.backend.configuration().dt * 1.e-9)
        self.assertEqual(self.sysinfo.dtm, self.backend.configuration().dtm * 1.e-9)
        self.assertEqual(self.sysinfo.basis_gates, self.backend.configuration().basis_gates)
        self.assertEqual(self.sysinfo.buffer, self.backend.defaults().buffer)

    def test_sample_rate(self):
        """Test that sample rate is 1/dt."""
        self.assertEqual(self.sysinfo.sample_rate, 1. / self.sysinfo.dt)

    def test_coupling_map(self):
        """Test that the coupling map is returned and in the proper format."""
        self.assertEqual(self.sysinfo.coupling_map, {0: {1}})

    def test_meas_map(self):
        """Test getting the measure map."""
        self.assertEqual(self.sysinfo.meas_map, self.backend.configuration().meas_map)

    def test_hamiltonian(self):
        """Test the hamiltonian method."""
        self.assertEqual(self.sysinfo.hamiltonian(),
                         self.backend.configuration().hamiltonian['h_latex'])

    def test_freq_est(self):
        """Test extracting qubit frequencies."""
        self.assertEqual(self.sysinfo.qubit_freq_est(1),
                         self.backend.defaults().qubit_freq_est[1] * 1e9)
        self.assertEqual(self.sysinfo.meas_freq_est(0),
                         self.backend.defaults().meas_freq_est[0] * 1e9)

    def test_get_channels(self):
        """Test requesting channels from the system."""
        self.assertEqual(self.sysinfo.drives(0), DriveChannel(0))
        self.assertEqual(self.sysinfo.measures(1), MeasureChannel(1))

    def test_get_property(self):
        """Test extracting properties from the backend with SystemInfo."""
        self.assertEqual(self.sysinfo.get_property('backend_version'),
                         self.backend.properties().backend_version)
        self.assertEqual(self.sysinfo.get_property('qubits', 0, 'T1')[0],
                         self.backend.properties().qubits[0][0].value * 1e-6)
        self.assertEqual(self.sysinfo.get_property('qubits', 0, 'frequency')[0],
                         self.backend.properties().qubits[0][1].value * 1e6)

    def test_missing_properties(self):
        """Test that missing properties and overspecified properties return None."""
        self.assertEqual(self.sysinfo.get_property('DNE'), None)
        self.assertEqual(self.sysinfo.get_property('backend_version', 1), None)
        self.assertEqual(self.sysinfo.get_property('gates', 'missing'), None)
        self.assertEqual(self.sysinfo.get_property('gates', 'u1', 'missing'), None)

    def test_gate_error(self):
        """Test getting the gate errors."""
        self.assertEqual(self.sysinfo.gate_error('u1', 0),
                         self.backend.properties().gates[0].parameters[0].value)
        self.assertEqual(self.sysinfo.gate_error('u1', [0]),
                         self.backend.properties().gates[0].parameters[0].value)
        self.assertEqual(self.sysinfo.gate_error('cx', [0, 1]),
                         self.backend.properties().gates[3].parameters[0].value)

    def test_gate_length(self):
        """Test getting the gate duration."""
        self.assertEqual(self.sysinfo.gate_length('u1', 0),
                         self.backend.properties().gates[0].parameters[1].value * 1e-9)
        self.assertEqual(self.sysinfo.gate_length('u3', qubits=[0]),
                         self.backend.properties().gates[1].parameters[1].value * 1e-9)

    def test_ops(self):
        """Test `ops`."""
        self.assertEqual(self.sysinfo.ops, ['u1', 'u3', 'cx', 'measure'])

    def test_op_qubits(self):
        """Test `op_qubits`."""
        self.assertEqual(self.sysinfo.op_qubits('u1'), [0])
        self.assertEqual(self.sysinfo.op_qubits('u3'), [0, 1])
        self.assertEqual(self.sysinfo.op_qubits('cx'), [(0, 1)])
        self.assertEqual(self.sysinfo.op_qubits('measure'), [(0, 1)])

    def test_qubit_ops(self):
        """Test `qubit_ops`."""
        self.assertEqual(self.sysinfo.qubit_ops(0), ['u1', 'u3'])
        self.assertEqual(self.sysinfo.qubit_ops(1), ['u3'])
        self.assertEqual(self.sysinfo.qubit_ops((0, 1)), ['cx', 'measure'])

    def test_get_op(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.sysinfo.drives(0)))
        sysinfo = SystemInfo()
        sysinfo.add_op('tmp', 1, sched)
        sysinfo.add_op('tmp', 0, sched)
        self.assertEqual(sched.instructions, sysinfo.get('tmp', (0,)).instructions)

        self.assertIn('tmp', sysinfo.ops)
        self.assertEqual(sysinfo.op_qubits('tmp'), [0, 1])

    def test_remove(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.sysinfo.drives(0)))
        sysinfo = SystemInfo()
        sysinfo.add_op('tmp', 0, sched)
        sysinfo.remove_op('tmp', 0)
        self.assertFalse(sysinfo.has('tmp', 0))
        with self.assertRaises(PulseError):
            sysinfo.remove_op('not_there', (0,))

    def test_parameterized_schedule(self):
        """Test adding parameterized schedule."""
        sysinfo = SystemInfo()
        converter = QobjToInstructionConverter([], buffer=0)
        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P2*cos(np.pi*P1)')
        converted_instruction = converter(qobj)

        sysinfo.add_op('pv_test', 0, converted_instruction)
        self.assertEqual(sysinfo.get_parameters('pv_test', 0), ('P1', 'P2'))

        sched = sysinfo.get('pv_test', 0, P1=0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)
        with self.assertRaises(PulseError):
            sysinfo.get('pv_test', 0, 0, P1=-1)
        with self.assertRaises(PulseError):
            sysinfo.get('pv_test', 0, P1=1, P2=2, P3=3)

    def test_sequenced_parameterized_schedule(self):
        """Test parametrized schedule consists of multiple instruction. """
        sysinfo = SystemInfo()
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        sysinfo.add_op('inst_seq', 0, ParameterizedSchedule(*converted_instruction,
                                                            name='inst_seq'))

        with self.assertRaises(PulseError):
            sysinfo.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            sysinfo.get('inst_seq', 0, P1=1)

        with self.assertRaises(PulseError):
            sysinfo.get('inst_seq', 0, 1, 2, 3, P1=1)

        sched = sysinfo.get('inst_seq', 0, 1, 2, 3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = sysinfo.get('inst_seq', 0, P1=1, P2=2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = sysinfo.get('inst_seq', 0, 1, 2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

    def test_build_cmd_def(self):
        """Test building of cmd_def is properly built from backend."""
        self.assertTrue(self.sysinfo.has('u1', (0,)))
        self.assertTrue(self.sysinfo.has('u3', (0,)))
        self.assertTrue(self.sysinfo.has('u3', 1))
        self.assertTrue(self.sysinfo.has('cx', (0, 1)))
        self.assertEqual(self.sysinfo.get_parameters('u1', 0), ('P1',))
        u1_minus_pi = self.sysinfo.get('u1', 0, P1=1)
        fc_cmd = u1_minus_pi.instructions[0][-1].command
        self.assertEqual(fc_cmd.phase, -np.pi)
        for chan in u1_minus_pi.channels:
            self.assertEqual(chan.buffer, self.backend.defaults().buffer)

    def test_str_and_repr(self):
        """Test that the __str__ and __repr__ methods work."""
        self.assertEqual(
            str(self.sysinfo),
            "fake_openpulse_2q(2 qubits operating on ['u1', 'u2', 'u3', 'cx', 'id'])")
        self.assertEqual(
            repr(self.sysinfo),
            "SystemInfo(fake_openpulse_2q 2Q\n    Operations:\n{'u1': dict_keys([(0,)]),"
            " 'u3': dict_keys([(0,), (1,)]), 'cx': dict_keys([(0, 1)]), 'measure': dict_"
            "keys([(0, 1)])}\n    Properties:\n['backend_name', 'backend_version', 'last"
            "_update_date', 'qubits', 'gates', 'general']\n    Configuration:\n['n_uchan"
            "nels', 'u_channel_lo', 'meas_levels', 'qubit_lo_range', 'meas_lo_range', 'd"
            "t', 'dtm', 'rep_times', 'meas_kernels', 'discriminators', 'backend_name', '"
            "backend_version', 'n_qubits', 'basis_gates', 'gates', 'local', 'simulator',"
            " 'conditional', 'open_pulse', 'memory', 'max_shots', 'coupling_map', 'n_reg"
            "isters', 'meas_level', 'meas_map', 'channel_bandwidth', 'acquisition_latenc"
            "y', 'conditional_latency', 'hamiltonian']\n    Hamiltonian:\nNone)")
