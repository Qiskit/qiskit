# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests PassManagerConfig"""

from qiskit import QuantumRegister
from qiskit.providers.backend import Backend
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeMelbourne, FakeArmonk, FakeHanoi, FakeHanoiV2
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager_config import PassManagerConfig


class TestPassManagerConfig(QiskitTestCase):
    """Test PassManagerConfig.from_backend()."""

    def test_config_from_backend(self):
        """Test from_backend() with a valid backend.

        `FakeHanoi` is used in this testcase. This backend has `defaults` attribute
        that contains an instruction schedule map.
        """
        backend = FakeHanoi()
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertEqual(config.inst_map, backend.defaults().instruction_schedule_map)
        self.assertEqual(
            str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map))
        )

    def test_config_from_backend_v2(self):
        """Test from_backend() with a BackendV2 instance."""
        backend = FakeHanoiV2()
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.operation_names)
        self.assertEqual(config.inst_map, backend.instruction_schedule_map)
        self.assertEqual(config.coupling_map.get_edges(), backend.coupling_map.get_edges())

    def test_invalid_backend(self):
        """Test from_backend() with an invalid backend."""
        with self.assertRaises(AttributeError):
            PassManagerConfig.from_backend(Backend())

    def test_from_backend_and_user(self):
        """Test from_backend() with a backend and user options.

        `FakeMelbourne` is used in this testcase. This backend does not have
        `defaults` attribute and thus not provide an instruction schedule map.
        """
        qr = QuantumRegister(4, "qr")
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        backend = FakeMelbourne()
        config = PassManagerConfig.from_backend(
            backend, basis_gates=["user_gate"], initial_layout=initial_layout
        )
        self.assertEqual(config.basis_gates, ["user_gate"])
        self.assertNotEqual(config.basis_gates, backend.configuration().basis_gates)
        self.assertIsNone(config.inst_map)
        self.assertEqual(
            str(config.coupling_map), str(CouplingMap(backend.configuration().coupling_map))
        )
        self.assertEqual(config.initial_layout, initial_layout)

    def test_from_backendv1_inst_map_is_none(self):
        """Test that from_backend() works with backend that has defaults defined as None."""
        backend = FakeHanoi()
        backend.defaults = lambda: None
        config = PassManagerConfig.from_backend(backend)
        self.assertIsInstance(config, PassManagerConfig)
        self.assertIsNone(config.inst_map)

    def test_simulator_backend_v1(self):
        """Test that from_backend() works with backendv1 simulator."""
        backend = QasmSimulatorPy()
        config = PassManagerConfig.from_backend(backend)
        self.assertIsInstance(config, PassManagerConfig)
        self.assertIsNone(config.inst_map)
        self.assertIsNone(config.coupling_map)

    def test_invalid_user_option(self):
        """Test from_backend() with an invalid user option."""
        with self.assertRaises(TypeError):
            PassManagerConfig.from_backend(FakeMelbourne(), invalid_option=None)

    def test_str(self):
        """Test string output."""
        pm_config = PassManagerConfig.from_backend(FakeArmonk())
        # For testing remove instruction schedule map it's str output is non-deterministic
        # based on hash seed
        pm_config.inst_map = None
        str_out = str(pm_config)
        expected = """Pass Manager Config:
	initial_layout: None
	basis_gates: ['id', 'rz', 'sx', 'x']
	inst_map: None
	coupling_map: None
	layout_method: None
	routing_method: None
	translation_method: None
	scheduling_method: None
	instruction_durations: id(0,): 7.111111111111111e-08 s
	rz(0,): 0.0 s
	sx(0,): 7.111111111111111e-08 s
	x(0,): 7.111111111111111e-08 s
	measure(0,): 4.977777777777777e-06 s
	
	backend_properties: {'backend_name': 'ibmq_armonk',
	 'backend_version': '2.4.3',
	 'gates': [{'gate': 'id',
	            'name': 'id0',
	            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 38, 15, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_error',
	                            'unit': '',
	                            'value': 0.00019769550670970334},
	                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_length',
	                            'unit': 'ns',
	                            'value': 71.11111111111111}],
	            'qubits': [0]},
	           {'gate': 'rz',
	            'name': 'rz0',
	            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_error',
	                            'unit': '',
	                            'value': 0},
	                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_length',
	                            'unit': 'ns',
	                            'value': 0}],
	            'qubits': [0]},
	           {'gate': 'sx',
	            'name': 'sx0',
	            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 38, 15, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_error',
	                            'unit': '',
	                            'value': 0.00019769550670970334},
	                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_length',
	                            'unit': 'ns',
	                            'value': 71.11111111111111}],
	            'qubits': [0]},
	           {'gate': 'x',
	            'name': 'x0',
	            'parameters': [{'date': datetime.datetime(2021, 3, 15, 0, 38, 15, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_error',
	                            'unit': '',
	                            'value': 0.00019769550670970334},
	                           {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	                            'name': 'gate_length',
	                            'unit': 'ns',
	                            'value': 71.11111111111111}],
	            'qubits': [0]}],
	 'general': [],
	 'last_update_date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	 'qubits': [[{'date': datetime.datetime(2021, 3, 15, 0, 36, 17, tzinfo=tzoffset(None, -14400)),
	              'name': 'T1',
	              'unit': 'us',
	              'value': 182.6611165336624},
	             {'date': datetime.datetime(2021, 3, 14, 0, 33, 45, tzinfo=tzoffset(None, -18000)),
	              'name': 'T2',
	              'unit': 'us',
	              'value': 237.8589220110257},
	             {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	              'name': 'frequency',
	              'unit': 'GHz',
	              'value': 4.971852852405576},
	             {'date': datetime.datetime(2021, 3, 15, 0, 40, 24, tzinfo=tzoffset(None, -14400)),
	              'name': 'anharmonicity',
	              'unit': 'GHz',
	              'value': -0.34719293148282626},
	             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),
	              'name': 'readout_error',
	              'unit': '',
	              'value': 0.02400000000000002},
	             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),
	              'name': 'prob_meas0_prep1',
	              'unit': '',
	              'value': 0.0234},
	             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),
	              'name': 'prob_meas1_prep0',
	              'unit': '',
	              'value': 0.024599999999999955},
	             {'date': datetime.datetime(2021, 3, 15, 0, 35, 20, tzinfo=tzoffset(None, -14400)),
	              'name': 'readout_length',
	              'unit': 'ns',
	              'value': 4977.777777777777}]]}
	approximation_degree: None
	seed_transpiler: None
	timing_constraints: None
	unitary_synthesis_method: default
	unitary_synthesis_plugin_config: None
	target: None
"""
        self.assertEqual(str_out, expected)
