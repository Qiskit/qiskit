# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

import datetime
import itertools
import operator
import unittest

from test import combine
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.execute_function import execute
from qiskit.test.base import QiskitTestCase
from qiskit.providers.fake_provider import (
    FakeProviderForBackendV2,
    FakeProvider,
    FakeMumbaiV2,
    FakeYorktown,
    FakeMumbai,
    FakeWashington,
    FakeSherbrooke,
    FakePrague,
)
from qiskit.providers.backend_compat import BackendV2Converter
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.backend import BackendV2
from qiskit.utils import optionals
from qiskit.circuit.library import (
    SXGate,
    MCPhaseGate,
    MCXGate,
    RZGate,
    RXGate,
    U2Gate,
    U1Gate,
    U3Gate,
    YGate,
    ZGate,
    PauliGate,
    SwapGate,
    RGate,
    MCXGrayCode,
    RYGate,
    CZGate,
    ECRGate,
    UnitaryGate,
    UCGate,
    Initialize,
    DiagonalGate,
)
from qiskit.circuit import ControlledGate, Parameter
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    ContinueLoopOp,
    BreakLoopOp,
    SwitchCaseOp,
)

FAKE_PROVIDER_FOR_BACKEND_V2 = FakeProviderForBackendV2()
FAKE_PROVIDER = FakeProvider()


@ddt
class TestFakeBackends(QiskitTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.circuit = QuantumCircuit(2)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.x(0)
        cls.circuit.x(1)
        cls.circuit.measure_all()

    @combine(
        backend=[be for be in FAKE_PROVIDER_FOR_BACKEND_V2.backends() if be.num_qubits > 1],
        optimization_level=[0, 1, 2, 3],
    )
    def test_circuit_on_fake_backend_v2(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.num_qubits > 20:
            self.skipTest("Unable to run fake_backend %s without qiskit-aer" % backend.backend_name)
        job = execute(
            self.circuit,
            backend,
            optimization_level=optimization_level,
            seed_simulator=42,
            seed_transpiler=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    @combine(
        backend=[be for be in FAKE_PROVIDER.backends() if be.configuration().num_qubits > 1],
        optimization_level=[0, 1, 2, 3],
        dsc="Test execution path on {backend} with optimization level {optimization_level}",
        name="{backend}_opt_level_{optimization_level}",
    )
    def test_circuit_on_fake_backend(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.configuration().num_qubits > 20:
            self.skipTest(
                "Unable to run fake_backend %s without qiskit-aer"
                % backend.configuration().backend_name
            )
        job = execute(
            self.circuit,
            backend,
            optimization_level=optimization_level,
            seed_simulator=42,
            seed_transpiler=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_qobj_failure(self):
        backend = FAKE_PROVIDER.backends()[-1]
        tqc = transpile(self.circuit, backend)
        qobj = assemble(tqc, backend)
        with self.assertRaises(QiskitError):
            backend.run(qobj)

    @data(*FAKE_PROVIDER.backends())
    def test_to_dict_properties(self, backend):
        properties = backend.properties()
        if properties:
            self.assertIsInstance(backend.properties().to_dict(), dict)
        else:
            self.assertTrue(backend.configuration().simulator)

    @data(*FAKE_PROVIDER_FOR_BACKEND_V2.backends())
    def test_convert_to_target(self, backend):
        target = backend.target
        if target.dt is not None:
            self.assertLess(target.dt, 1e-6)

    @data(*FAKE_PROVIDER_FOR_BACKEND_V2.backends())
    def test_backend_v2_dtm(self, backend):
        if backend.dtm:
            self.assertLess(backend.dtm, 1e-6)

    @data(*FAKE_PROVIDER.backends())
    def test_to_dict_configuration(self, backend):
        configuration = backend.configuration()
        if configuration.open_pulse:
            self.assertLess(configuration.dt, 1e-6)
            self.assertLess(configuration.dtm, 1e-6)
            for i in configuration.qubit_lo_range:
                self.assertGreater(i[0], 1e6)
                self.assertGreater(i[1], 1e6)
                self.assertLess(i[0], i[1])

            for i in configuration.meas_lo_range:
                self.assertGreater(i[0], 1e6)
                self.assertGreater(i[0], 1e6)
                self.assertLess(i[0], i[1])

            for i in configuration.rep_times:
                self.assertGreater(i, 0)
                self.assertLess(i, 1)

        self.assertIsInstance(configuration.to_dict(), dict)

    @data(*FAKE_PROVIDER.backends())
    def test_defaults_to_dict(self, backend):
        if hasattr(backend, "defaults"):
            defaults = backend.defaults()
            self.assertIsInstance(defaults.to_dict(), dict)

            for i in defaults.qubit_freq_est:
                self.assertGreater(i, 1e6)
                self.assertGreater(i, 1e6)

            for i in defaults.meas_freq_est:
                self.assertGreater(i, 1e6)
                self.assertGreater(i, 1e6)
        else:
            self.skipTest("Backend %s does not have defaults" % backend)

    def test_delay_circuit(self):
        backend = FakeMumbaiV2()
        qc = QuantumCircuit(2)
        qc.delay(502, 0, unit="ns")
        qc.x(1)
        qc.delay(250, 1, unit="ns")
        qc.measure_all()
        res = transpile(qc, backend)
        self.assertIn("delay", res.count_ops())

    @data(0, 1, 2, 3)
    def test_converter(self, opt_level):
        backend = FakeYorktown()
        backend_v2 = BackendV2Converter(backend)
        self.assertIsInstance(backend_v2, BackendV2)
        res = transpile(self.circuit, backend_v2, optimization_level=opt_level)
        job = backend_v2.run(res)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_converter_delay_circuit(self):
        backend = FakeMumbai()
        backend_v2 = BackendV2Converter(backend, add_delay=True)
        self.assertIsInstance(backend_v2, BackendV2)
        qc = QuantumCircuit(2)
        qc.delay(502, 0, unit="ns")
        qc.x(1)
        qc.delay(250, 1, unit="ns")
        qc.measure_all()
        res = transpile(qc, backend_v2)
        self.assertIn("delay", res.count_ops())

    def test_non_cx_tests(self):
        backend = FakePrague()
        self.assertIsInstance(backend.target.operation_from_name("cz"), CZGate)
        backend = FakeSherbrooke()
        self.assertIsInstance(backend.target.operation_from_name("ecr"), ECRGate)

    @unittest.skipUnless(optionals.HAS_AER, "Aer required for this test")
    def test_converter_simulator(self):
        class MCSXGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcsx",
                    1 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=SXGate(),
                )

        class MCYGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcy",
                    1 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=YGate(),
                )

        class MCZGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcz",
                    1 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=ZGate(),
                )

        class MCRXGate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcrx",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RXGate(theta),
                )

        class MCRYGate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcry",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RYGate(theta),
                )

        class MCRZGate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcrz",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RZGate(theta),
                )

        class MCRGate(ControlledGate):
            def __init__(self, theta, phi, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcr",
                    1 + num_ctrl_qubits,
                    [theta, phi],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RGate(theta, phi),
                )

        class MCU1Gate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu1",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U1Gate(theta),
                )

        class MCU2Gate(ControlledGate):
            def __init__(self, theta, lam, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu2",
                    1 + num_ctrl_qubits,
                    [theta, lam],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U2Gate(theta, lam),
                )

        class MCU3Gate(ControlledGate):
            def __init__(self, theta, lam, phi, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu3",
                    1 + num_ctrl_qubits,
                    [theta, phi, lam],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U3Gate(theta, phi, lam),
                )

        class MCUGate(ControlledGate):
            def __init__(self, theta, lam, phi, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu",
                    1 + num_ctrl_qubits,
                    [theta, phi, lam],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U3Gate(theta, phi, lam),
                )

        class MCSwapGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcswap",
                    2 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=SwapGate(),
                )

        from qiskit_aer import AerSimulator
        from qiskit_aer.library import (
            SaveExpectationValue,
            SaveAmplitudes,
            SaveStatevectorDict,
            SaveSuperOp,
            SaveClifford,
            SaveMatrixProductState,
            SaveDensityMatrix,
            SaveProbabilities,
            SaveStatevector,
            SetDensityMatrix,
            SetUnitary,
            SaveState,
            SetMatrixProductState,
            SaveUnitary,
            SetSuperOp,
            SaveExpectationValueVariance,
            SaveStabilizer,
            SetStatevector,
            SetStabilizer,
            SaveAmplitudesSquared,
            SaveProbabilitiesDict,
        )
        from qiskit_aer.noise.errors import ReadoutError
        from qiskit_aer.noise.noise_model import QuantumErrorLocation

        sim = AerSimulator()
        phi = Parameter("phi")
        lam = Parameter("lam")
        backend = BackendV2Converter(
            sim,
            name_mapping={
                "mcsx": MCSXGate,
                "mcp": MCPhaseGate,
                "mcphase": MCPhaseGate,
                "quantum_channel": QuantumChannel,
                "initialize": Initialize,
                "save_expval": SaveExpectationValue,
                "diagonal": DiagonalGate,
                "save_amplitudes": SaveAmplitudes,
                "roerror": ReadoutError,
                "mcrx": MCRXGate,
                "kraus": Kraus,
                "save_statevector_dict": SaveStatevectorDict,
                "mcx": MCXGate,
                "mcu1": MCU1Gate,
                "mcu2": MCU2Gate,
                "mcu3": MCU3Gate,
                "save_superop": SaveSuperOp,
                "multiplexer": UCGate,
                "mcy": MCYGate,
                "superop": SuperOp,
                "save_clifford": SaveClifford,
                "save_matrix_product_state": SaveMatrixProductState,
                "save_density_matrix": SaveDensityMatrix,
                "save_probabilities": SaveProbabilities,
                "if_else": IfElseOp,
                "while_loop": WhileLoopOp,
                "for_loop": ForLoopOp,
                "switch_case": SwitchCaseOp,
                "break_loop": BreakLoopOp,
                "continue_loop": ContinueLoopOp,
                "save_statevector": SaveStatevector,
                "mcu": MCUGate,
                "set_density_matrix": SetDensityMatrix,
                "qerror_loc": QuantumErrorLocation,
                "unitary": UnitaryGate,
                "mcz": MCZGate,
                "pauli": PauliGate,
                "set_unitary": SetUnitary,
                "save_state": SaveState,
                "mcswap": MCSwapGate,
                "set_matrix_product_state": SetMatrixProductState,
                "save_unitary": SaveUnitary,
                "mcr": MCRGate,
                "mcx_gray": MCXGrayCode,
                "mcrz": MCRZGate,
                "set_superop": SetSuperOp,
                "save_expval_var": SaveExpectationValueVariance,
                "save_stabilizer": SaveStabilizer,
                "set_statevector": SetStatevector,
                "mcry": MCRYGate,
                "set_stabilizer": SetStabilizer,
                "save_amplitudes_sq": SaveAmplitudesSquared,
                "save_probabilities_dict": SaveProbabilitiesDict,
                "cu2": U2Gate(phi, lam).control(),
            },
        )
        self.assertIsInstance(backend, BackendV2)
        res = transpile(self.circuit, backend)
        job = backend.run(res)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_filter_faulty_qubits_backend_v2_converter(self):
        """Test faulty qubits in v2 conversion."""
        backend = FakeWashington()
        # Get properties dict to make it easier to work with the properties API
        # is difficult to edit because of the multiple layers of nesting and
        # different object types
        props_dict = backend.properties().to_dict()
        for i in range(62, 67):
            non_operational = {
                "date": datetime.datetime.utcnow(),
                "name": "operational",
                "unit": "",
                "value": 0,
            }
            props_dict["qubits"][i].append(non_operational)
        backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(62, 67):
            for qarg in v2_backend.target.qargs:
                self.assertNotIn(i, qarg)

    def test_filter_faulty_qubits_backend_v2_converter_with_delay(self):
        """Test faulty qubits in v2 conversion."""
        backend = FakeWashington()
        # Get properties dict to make it easier to work with the properties API
        # is difficult to edit because of the multiple layers of nesting and
        # different object types
        props_dict = backend.properties().to_dict()
        for i in range(62, 67):
            non_operational = {
                "date": datetime.datetime.utcnow(),
                "name": "operational",
                "unit": "",
                "value": 0,
            }
            props_dict["qubits"][i].append(non_operational)
        backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True, add_delay=True)
        for i in range(62, 67):
            for qarg in v2_backend.target.qargs:
                self.assertNotIn(i, qarg)

    def test_filter_faulty_qubits_and_gates_backend_v2_converter(self):
        """Test faulty gates and qubits."""
        backend = FakeWashington()
        # Get properties dict to make it easier to work with the properties API
        # is difficult to edit because of the multiple layers of nesting and
        # different object types
        props_dict = backend.properties().to_dict()
        for i in range(62, 67):
            non_operational = {
                "date": datetime.datetime.utcnow(),
                "name": "operational",
                "unit": "",
                "value": 0,
            }
            props_dict["qubits"][i].append(non_operational)
        invalid_cx_edges = {
            (113, 114),
            (114, 113),
            (96, 100),
            (100, 96),
            (114, 109),
            (109, 114),
            (24, 34),
            (34, 24),
        }
        non_operational_gate = {
            "date": datetime.datetime.utcnow(),
            "name": "operational",
            "unit": "",
            "value": 0,
        }
        for gate in props_dict["gates"]:
            if tuple(gate["qubits"]) in invalid_cx_edges:
                gate["parameters"].append(non_operational_gate)

        backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(62, 67):
            for qarg in v2_backend.target.qargs:
                self.assertNotIn(i, qarg)
        for edge in invalid_cx_edges:
            self.assertNotIn(edge, v2_backend.target["cx"])

    def test_filter_faulty_gates_v2_converter(self):
        """Test just faulty gates in conversion."""
        backend = FakeWashington()
        # Get properties dict to make it easier to work with the properties API
        # is difficult to edit because of the multiple layers of nesting and
        # different object types
        props_dict = backend.properties().to_dict()
        invalid_cx_edges = {
            (113, 114),
            (114, 113),
            (96, 100),
            (100, 96),
            (114, 109),
            (109, 114),
            (24, 34),
            (34, 24),
        }
        non_operational_gate = {
            "date": datetime.datetime.utcnow(),
            "name": "operational",
            "unit": "",
            "value": 0,
        }
        for gate in props_dict["gates"]:
            if tuple(gate["qubits"]) in invalid_cx_edges:
                gate["parameters"].append(non_operational_gate)

        backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(62, 67):
            self.assertIn((i,), v2_backend.target.qargs)
        for edge in invalid_cx_edges:
            self.assertNotIn(edge, v2_backend.target["cx"])

    def test_filter_faulty_no_faults_v2_converter(self):
        """Test that faulty qubit filtering does nothing with all operational qubits and gates."""
        backend = FakeWashington()
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(v2_backend.num_qubits):
            self.assertIn((i,), v2_backend.target.qargs)

    @data(0, 1, 2, 3)
    def test_faulty_full_path_transpile_connected_cmap(self, opt_level):
        backend = FakeYorktown()
        non_operational_gate = {
            "date": datetime.datetime.utcnow(),
            "name": "operational",
            "unit": "",
            "value": 0,
        }
        props = backend.properties().to_dict()
        for gate in props["gates"]:
            if tuple(sorted(gate["qubits"])) == (0, 1):
                gate["parameters"].append(non_operational_gate)
        backend._properties = BackendProperties.from_dict(props)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        qc = QuantumCircuit(5)
        for x, y in itertools.product(range(5), range(5)):
            if x == y:
                continue
            qc.cx(x, y)
        tqc = transpile(qc, v2_backend, seed_transpiler=433, optimization_level=opt_level)
        connections = [tuple(sorted(tqc.find_bit(q).index for q in x.qubits)) for x in tqc.data]
        self.assertNotIn((0, 1), connections)
