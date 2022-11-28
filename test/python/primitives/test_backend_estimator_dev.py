# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BackendEstimator."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from unittest.mock import MagicMock, Mock, patch
from test import combine

from ddt import ddt, data, unpack
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import EstimatorResult
from qiskit.primitives.backend_estimator_dev import (
    AbelianDecomposer,
    BackendEstimator,
    NaiveDecomposer,
)
from qiskit.providers import BackendV2, Options, JobV1
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.result import Counts
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import ApplyLayout, SetLayout
from qiskit.test import QiskitTestCase

BACKENDS = [FakeNairobi(), FakeNairobiV2()]


################################################################################
## AUXILIARY
################################################################################
def measurement_circuit_examples() -> Iterator[tuple[list[str], QuantumCircuit]]:
    """Generator of commuting Paulis and corresponding measurement circuits.

    Yields:
        - List of commuting Pauli strings
        - Quantum circuit to measure said Paulis
    """
    I = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    I.measure(0, 0)
    yield ["I", "Z"], I

    X = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    X.h(0)
    X.measure(0, 0)
    yield ["X", "I"], X

    Y = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    Y.sdg(0)
    Y.h(0)
    Y.measure(0, 0)
    yield ["Y", "I"], Y

    Z = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    Z.measure(0, 0)
    yield ["Z", "I"], Z

    II = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    II.measure(0, 0)
    yield ["II"], II

    IY = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    IY.sdg(0)
    IY.h(0)
    IY.measure(0, 0)
    yield ["IY", "II"], IY

    XY = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    XY.h(1)
    XY.sdg(0)
    XY.h(0)
    XY.measure([0, 1], [0, 1])
    yield ["XY", "II", "XI", "IY"], XY

    XX = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    XX.h(1)
    XX.h(0)
    XX.measure([0, 1], [0, 1])
    yield ["XX", "IX", "XI", "II"], XX

    ZZ = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    ZZ.measure([0, 1], [0, 1])
    yield ["ZZ", "IZ", "ZI", "II"], ZZ

    XYZ = QuantumCircuit(3, 3)  # pylint: disable=invalid-name
    XYZ.h(2)
    XYZ.sdg(1)
    XYZ.h(1)
    XYZ.measure([0, 1, 2], [0, 1, 2])
    yield ["XYZ", "XII", "IYI", "IIZ", "XIZ", "III"], XYZ

    YIX = QuantumCircuit(3, 2)  # pylint: disable=invalid-name
    YIX.sdg(2)
    YIX.h(2)
    YIX.h(0)
    YIX.measure([0, 2], [0, 1])
    yield ["YIX", "IIX", "YII", "III"], YIX

    IXII = QuantumCircuit(4, 1)  # pylint: disable=invalid-name
    IXII.h(2)
    IXII.measure(2, 0)
    yield ["IXII", "IIII"], IXII


def build_composition_data(
    target_qubits: int, layout_intlist: Sequence[int], measured_qubits: Sequence[int]
) -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Build base and measurement circuits, and respective composition.

    Args:
        target_qubits: the number of qubits to target during transpilation
        layout_intlist: indices to map virtual qubits to during transpilaiton
        measured_qubits: virtual qubits to measure

    Returns:
        - Transpiled base circuit: with a `final_layout` entry in its metadata
        - Measurement circuit: before transpilation (i.e. no layout applied)
        - Transpiled measurement circuit: with `final_layout` applied
    """
    _, transpiled_base = build_base_circuit(target_qubits, layout_intlist)
    transpiled_base.metadata.update({"extra_base": Mock()})
    measurement, transpiled_measurement = build_measurement_circuit(
        target_qubits, layout_intlist, measured_qubits
    )
    measurement.metadata.update({"extra_meas": Mock()})
    return transpiled_base, measurement, transpiled_measurement


def build_base_circuit(target_qubits, layout_intlist):
    """Build example base and transpiled base circuits."""
    num_qubits = len(layout_intlist)
    base = QuantumCircuit(num_qubits)
    base.h(range(num_qubits))  # Dummy single-qubit gates
    if num_qubits > 1:
        base.cx(range(-1, num_qubits - 1), range(num_qubits))  # Dummy two-qubit gates
    transpiled_base = transpile_to_layout(base, target_qubits, layout_intlist)
    return base, transpiled_base


def build_measurement_circuit(target_qubits, layout_intlist, measured_qubits):
    """Build example measurement and transpiled measurement circuits."""
    num_qubits = len(layout_intlist)
    num_measurements = len(measured_qubits)
    measurement = QuantumCircuit(num_qubits, num_measurements)
    measurement.h(measured_qubits)  # Dummy gates (i.e. X measurements)
    measurement.measure(measured_qubits, range(num_measurements))
    measurement.metadata = {"measured_qubit_indices": measured_qubits}
    transpiled_measurement = transpile_to_layout(measurement, target_qubits, layout_intlist)
    return measurement, transpiled_measurement


def transpile_to_layout(circuit, target_qubits, layout_intlist):
    """Transpile circuit to match a given layout intlist."""
    if circuit.num_qubits != len(layout_intlist):
        raise ValueError("Circuit incompatible with requested layout.")
    if circuit.num_qubits > target_qubits:
        raise ValueError("Circuit incompatible with requested target.")
    layout_dict = dict.fromkeys(range(target_qubits))
    layout_dict.update(dict(zip(layout_intlist, circuit.qubits)))
    applied_layout = Layout(layout_dict)
    passes = [SetLayout(layout=applied_layout), ApplyLayout()]
    pass_manager = PassManager(passes=passes)
    transpiled = pass_manager.run(circuit)
    transpiled.metadata = {"final_layout": applied_layout}
    return transpiled


################################################################################
## UNIT TESTS
################################################################################
@ddt
class TestTranspilation(QiskitTestCase):
    """Test transpilation logic."""

    @data(
        [3, (0, 1, 2)],
        [4, (0, 1, 2, 3)],
        [4, (1, 3, 2, 0)],
        [4, (0, 1, 3)],
        [4, (3, 1)],
    )
    @unpack
    def test_transpile(self, target_qubits, layout_intlist):
        """Test transpile functionality.

        Assumptions for final layout inferrence:
            - Circuits passed to Qiskit-Terra's `transpile` are all measured
            - Measurements are in order (i.e. coming from `measure_all()`)
            - Classical bits remain in order in measurements through transpilation
        """
        # Input and measured circuits
        num_qubits = len(layout_intlist)
        input_circuit = QuantumCircuit(num_qubits)
        measured_circuit = input_circuit.copy()
        measured_circuit.measure_all()
        # Transpiled circuit (only changes layout and num of qubits)
        layout_dict = dict.fromkeys(range(target_qubits))
        layout_dict.update(dict(zip(layout_intlist, input_circuit.qubits)))
        applied_layout = Layout(layout_dict)
        passes = [SetLayout(layout=applied_layout), ApplyLayout()]
        pass_manager = PassManager(passes=passes)
        transpiled_circuit = pass_manager.run(measured_circuit)  # TODO: skip_transpilation
        # Test patching terra's transpile call
        backend = Mock(BackendV2)
        estimator = BackendEstimator(backend)
        estimator._transpile_options = MagicMock(Options)
        with patch("qiskit.primitives.backend_estimator_dev.transpile", spec=True) as mock:
            mock.return_value = transpiled_circuit
            output_circuit = estimator._transpile(input_circuit)
        mock.assert_called_once()
        (call_circuit, call_backend), call_kwargs = mock.call_args
        self.assertEqual(call_circuit, measured_circuit)
        self.assertIs(call_backend, backend)
        self.assertEqual(call_kwargs, estimator._transpile_options.__dict__)
        self.assertEqual(output_circuit, transpiled_circuit)
        self.assertIsInstance(output_circuit, QuantumCircuit)
        inferred_layout = output_circuit.metadata.get("final_layout")
        self.assertEqual(inferred_layout, applied_layout)
        self.assertIsInstance(inferred_layout, Layout)

    def test_run_bound_pass_manager(self):
        """Test bound pass manager runs."""
        backend = Mock(BackendV2)
        estimator = BackendEstimator(backend)
        # Invalid input
        self.assertRaises(TypeError, estimator._run_bound_pass_manager, "circuit")
        # No pass manager
        circuit = Mock(QuantumCircuit)
        self.assertIs(circuit, estimator._run_bound_pass_manager(circuit))
        # Pass manager runs
        mock_circuit = Mock(QuantumCircuit)
        estimator._bound_pass_manager = Mock(PassManager)
        estimator._bound_pass_manager.run.return_value = mock_circuit
        self.assertIs(mock_circuit, estimator._run_bound_pass_manager(circuit))
        estimator._bound_pass_manager.run.assert_called_once_with(circuit)


@ddt
class TestMeasurement(QiskitTestCase):
    """Test measurement logic."""

    def test_observable_decomposer(self):
        """Test observable decomposer property."""
        estimator = BackendEstimator(Mock(BackendV2))
        self.assertTrue(estimator.abelian_grouping)
        self.assertIsInstance(estimator._observable_decomposer, AbelianDecomposer)
        self.assertIsNot(estimator._observable_decomposer, estimator._observable_decomposer)
        estimator.abelian_grouping = False
        self.assertFalse(estimator.abelian_grouping)
        self.assertIsInstance(estimator._observable_decomposer, NaiveDecomposer)
        self.assertIsNot(estimator._observable_decomposer, estimator._observable_decomposer)

    @data(*measurement_circuit_examples())
    @unpack
    def test_build_single_measurement_circuit(self, paulis, measurement):
        """Test measurement circuits for a given observable."""
        # Preparation
        observable = SparsePauliOp(paulis)  # TODO: custom coeffs
        coeffs = tuple(np.real_if_close(observable.coeffs).tolist())
        qubit_index_map = {qubit: i for i, qubit in enumerate(measurement.qubits)}
        meas_indices = tuple(
            qubit_index_map[qargs[0]] for inst, qargs, _ in measurement if inst.name == "measure"
        )
        paulis = PauliList.from_symplectic(
            observable.paulis.z[:, meas_indices],
            observable.paulis.x[:, meas_indices],
            observable.paulis.phase,
        )
        # Tests
        circuit = BackendEstimator(Mock(BackendV2))._build_single_measurement_circuit(observable)
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit, measurement)
        self.assertIsInstance(circuit.metadata.get("measured_qubit_indices"), tuple)
        self.assertEqual(circuit.metadata.get("measured_qubit_indices"), meas_indices)
        self.assertIsInstance(circuit.metadata.get("paulis"), PauliList)
        self.assertEqual(circuit.metadata.get("paulis"), paulis)
        self.assertIsInstance(circuit.metadata.get("coeffs"), tuple)
        self.assertEqual(circuit.metadata.get("coeffs"), coeffs)


@ddt
class TestComposition(QiskitTestCase):
    """Test composition logic."""

    @data(
        build_composition_data(2, (0,), (0,)),
        build_composition_data(2, (1,), (0,)),
        build_composition_data(2, (1, 0), (1,)),
        build_composition_data(2, (1, 0), (0, 1)),
        build_composition_data(3, (2,), (0,)),
        build_composition_data(3, (2, 0), (0, 1)),
        build_composition_data(3, (1, 2), (1,)),
        build_composition_data(3, (1, 0, 2), (1, 2)),
        build_composition_data(4, (0, 1, 2), (0, 1)),
        build_composition_data(4, (1, 0, 3), (1, 2)),
        build_composition_data(4, (2, 1, 3, 0), (0, 1, 2)),
        build_composition_data(4, (0, 2, 1, 3), (0, 1, 2, 3)),
    )
    @unpack
    def test_compose_single_measurement(self, transpiled_base, measurement, transpiled_measurement):
        """Test coposition of single base circuit and measurement pair."""
        # Preapration
        expected_composition = transpiled_base.compose(transpiled_measurement)
        expected_metadata = {**transpiled_base.metadata, **measurement.metadata}
        expected_metadata.pop("measured_qubit_indices")
        # Test
        backend = Mock(BackendV2)
        estimator = BackendEstimator(backend)
        with patch("qiskit.primitives.backend_estimator_dev.transpile", spec=True) as mock:
            mock.return_value = transpiled_measurement
            composition = estimator._compose_single_measurement(transpiled_base, measurement)
        mock.assert_called_once()
        (call_circuit, call_backend), call_kwargs = mock.call_args
        self.assertEqual(call_circuit, measurement)
        self.assertIs(call_backend, backend)
        transpile_options = {**estimator.transpile_options.__dict__}
        transpile_options.update({"initial_layout": expected_metadata.get("final_layout")})
        self.assertEqual(call_kwargs, transpile_options)
        self.assertIsInstance(composition, QuantumCircuit)
        self.assertEqual(composition, expected_composition)
        self.assertEqual(composition.metadata, expected_metadata)


@ddt
class TestComputation(QiskitTestCase):
    """Test calculation logic."""

    @data(
        [{"0": 100, "1": 0}, "I", (1, 0)],
        [{"0": 0, "1": 100}, "I", (1, 0)],
        [{"0": 50, "1": 50}, "I", (1, 0)],
        [{"0": 50, "1": 50}, "X", (0, 1)],
        [{"0": 50, "1": 50}, "Y", (0, 1)],
        [{"0": 50, "1": 50}, "Z", (0, 1)],
        [{"0": 100, "1": 0}, "Z", (1, 0)],
        [{"0": 0, "1": 100}, "Z", (-1, 0)],
        [{"0": 80, "1": 20}, "Z", (0.6, 0.64)],
        [{"0": 60, "1": 40}, "Z", (0.2, 0.96)],
        [{"0": 40, "1": 60}, "Z", (-0.2, 0.96)],
        [{"0": 20, "1": 80}, "Z", (-0.6, 0.64)],
        [{"00": 80, "11": 20}, "ZZ", (1, 0)],
        [{"00": 80, "10": 20}, "ZZ", (0.6, 0.64)],
        [{"00": 20, "10": 80}, "ZZ", (-0.6, 0.64)],
        [{"11": 80, "01": 20}, "ZZ", (0.6, 0.64)],
        [{"11": 20, "01": 80}, "ZZ", (-0.6, 0.64)],
        [{"00": 80, "11": 20}, "ZI", (0.6, 0.64)],
        [{"00": 80, "10": 20}, "ZI", (0.6, 0.64)],
        [{"00": 20, "10": 80}, "ZI", (-0.6, 0.64)],
        [{"11": 80, "01": 20}, "ZI", (-0.6, 0.64)],
        [{"11": 20, "01": 80}, "ZI", (0.6, 0.64)],
        [{"11": 20, "01": 80}, "II", (1, 0)],
    )
    @unpack
    def test_compute_expval_variance_pair(self, counts, pauli, expected):
        """Test expval-variance pairs."""
        counts = Counts(counts)
        pauli = Pauli(pauli)
        pair = BackendEstimator._compute_expval_variance_pair(counts, pauli)
        self.assertEqual(pair, expected)

    @data(
        ["II", "00", +1],
        ["II", "01", +1],
        ["II", "10", +1],
        ["II", "11", +1],
        ["IX", "00", +1],
        ["IX", "01", -1],
        ["IX", "10", +1],
        ["IX", "11", -1],
        ["XI", "00", +1],
        ["XI", "01", +1],
        ["XI", "10", -1],
        ["XI", "11", -1],
        ["XX", "00", +1],
        ["XX", "01", -1],
        ["XX", "10", -1],
        ["XX", "11", +1],
        ["YZ", "00", +1],
        ["XY", "01", -1],
        ["ZX", "10", -1],
        ["ZZ", "11", +1],
        ["IXYZ", "0010", -1],
        ["IXYZ", "1000", +1],
        ["IXYZ", "1100", -1],
        ["IXYZ", "1101", +1],
        ["IXYZ", "0101", +1],
    )
    @unpack
    def test_measurement_coefficient(self, pauli, bitstring, expected):
        """Test measurement coefficients."""
        pauli = Pauli(pauli)
        coeff = BackendEstimator._measurement_coefficient(bitstring, pauli)
        self.assertEqual(coeff, expected)

    @data(
        ["II", 0],
        ["IZ", 1],
        ["ZI", 2],
        ["ZZ", 3],
        ["ZX", 3],
        ["XY", 3],
        ["IIII", 0],
        ["IXII", 4],
    )
    @unpack
    def test_pauli_integer_masks(self, pauli, expected):
        """Test Paulis integer masks."""
        pauli = Pauli(pauli)
        int_mask = BackendEstimator._pauli_integer_mask(pauli)
        self.assertEqual(int_mask, expected)

    @data(
        [(), ""],
        [(False,), "0"],
        [(True,), "1"],
        [(False, False), "00"],
        [(False, True), "01"],
        [(True, False), "10"],
        [(True, True), "11"],
        [(False, True, True, False, True, False, False, True), "01101001"],
        [(False, False, True, True, False, False, True, True), "00110011"],
    )
    @unpack
    def test_bitstring_from_mask(self, mask, expected):
        """Test `_bitstring_from_mask()`."""
        bitstring = BackendEstimator._bitstring_from_mask(mask)
        self.assertEqual(bitstring, "0b" + expected)
        bitstring = BackendEstimator._bitstring_from_mask(mask, little_endian=True)
        self.assertEqual(bitstring, "0b" + "".join(reversed(expected)))

    @data(
        ["0", 0],
        ["1", 1],
        ["00", 0],
        ["01", 1],
        ["10", 1],
        ["11", 0],
        ["10101100", 0],
        ["01001010", 1],
    )
    @unpack
    def test_parity_bit(self, bitstring, expected):
        """Test even parity bit."""
        integer = int(bitstring, 2)
        even_bit = BackendEstimator._parity_bit(integer)
        odd_bit = BackendEstimator._parity_bit(integer, even=False)
        self.assertEqual(even_bit, expected)
        self.assertEqual(even_bit, int(not odd_bit))


@ddt
class TestObservableDecomposer(QiskitTestCase):
    """Test ObservableDecomposer strategies."""

    @data(
        [NaiveDecomposer(), SparsePauliOp("IXYZ"), (SparsePauliOp("IXYZ"),)],
        [
            NaiveDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI"]),
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ],
        [
            NaiveDecomposer(),
            SparsePauliOp(["IXYZ", "IXII"]),
            (SparsePauliOp("IXYZ"), SparsePauliOp("IXII")),
        ],
        [
            NaiveDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
            (
                SparsePauliOp("IXYZ"),
                SparsePauliOp("ZYXI"),
                SparsePauliOp("IXII"),
                SparsePauliOp("ZYII"),
            ),
        ],
        [AbelianDecomposer(), SparsePauliOp("IXYZ"), (SparsePauliOp("IXYZ"),)],
        [
            AbelianDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI"]),
            (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI")),
        ],
        [
            AbelianDecomposer(),
            SparsePauliOp(["IXYZ", "IXII"]),
            (SparsePauliOp(["IXYZ", "IXII"]),),
        ],
        [
            AbelianDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
            (SparsePauliOp(["IXYZ", "IXII"]), SparsePauliOp(["ZYXI", "ZYII"])),
        ],
    )
    @unpack
    def test_decompose(self, decomposer, observable, expected):
        """Test decompose in ObservableDecomposer strategies."""
        components = decomposer.decompose(observable)
        self.assertEqual(components, expected)

    @data(
        [NaiveDecomposer(), SparsePauliOp("IXYZ"), PauliList(Pauli("IXYZ"))],
        [
            NaiveDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI"]),
            PauliList([Pauli("IXYZ"), Pauli("ZYXI")]),
        ],
        [
            NaiveDecomposer(),
            SparsePauliOp(["IXYZ", "IXII"]),
            PauliList([Pauli("IXYZ"), Pauli("IXII")]),
        ],
        [
            NaiveDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
            PauliList([Pauli("IXYZ"), Pauli("ZYXI"), Pauli("IXII"), Pauli("ZYII")]),
        ],
        [AbelianDecomposer(), SparsePauliOp("IXYZ"), PauliList(Pauli("IXYZ"))],
        [
            AbelianDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI"]),
            PauliList([Pauli("IXYZ"), Pauli("ZYXI")]),
        ],
        [
            AbelianDecomposer(),
            SparsePauliOp(["IXYZ", "IXII"]),
            PauliList([Pauli("IXYZ")]),
        ],
        [
            AbelianDecomposer(),
            SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
            PauliList([Pauli("IXYZ"), Pauli("ZYXI")]),
        ],
    )
    @unpack
    def test_pauli_basis(self, decomposer, observable, expected):
        """Test Pauli basis in ObservableDecomposer strategies."""
        basis = decomposer.extract_pauli_basis(observable)
        self.assertEqual(basis, expected)


################################################################################
## INTEGRATION TESTS
################################################################################
@ddt
class TestBackendEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.0284380963435145, -1.284366511861733

        self.psi = (RealAmplitudes(num_qubits=2, reps=2), RealAmplitudes(num_qubits=2, reps=3))
        self.params = tuple(psi.parameters for psi in self.psi)
        self.hamiltonian = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        self.theta = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )

    @combine(backend=BACKENDS)
    def test_estimator_run(self, backend):
        """Test Estimator.run()"""
        backend.set_options(seed_simulator=123)
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        estimator = BackendEstimator(backend=backend)

        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        job = estimator.run([psi1], [hamiltonian1], [theta1])
        self.assertIsInstance(job, JobV1)
        result = job.result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.5555572817900956], rtol=0.5, atol=0.2)

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        result2 = estimator.run([psi2], [hamiltonian1], [theta2]).result()
        np.testing.assert_allclose(result2.values, [2.97797666], rtol=0.5, atol=0.2)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        result3 = estimator.run([psi1, psi1], [hamiltonian2, hamiltonian3], [theta1] * 2).result()
        np.testing.assert_allclose(result3.values, [-0.551653, 0.07535239], rtol=0.5, atol=0.2)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        result4 = estimator.run(
            [psi1, psi2, psi1], [hamiltonian1, hamiltonian2, hamiltonian3], [theta1, theta2, theta3]
        ).result()
        np.testing.assert_allclose(
            result4.values, [1.55555728, 0.17849238, -1.08766318], rtol=0.5, atol=0.2
        )

    @combine(backend=BACKENDS)
    def test_estimator_run_no_params(self, backend):
        """test for estimator without parameters"""
        backend.set_options(seed_simulator=123)
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        est = BackendEstimator(backend=backend)
        result = est.run([circuit], [self.observable]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733], rtol=0.05)

    @combine(backend=BACKENDS)
    def test_run_1qubit(self, backend):
        """Test for 1-qubit cases"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = BackendEstimator(backend=backend)
        result = est.run([qc], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_run_2qubits(self, backend):
        """Test for 2-qubit cases (to check endian)"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = BackendEstimator(backend=backend)
        result = est.run([qc], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op2], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc], [op3], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op3], [[]]).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_run_errors(self, backend):
        """Test for errors"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = BackendEstimator(backend=backend)
        with self.assertRaises(ValueError):
            est.run([qc], [op2], [[]]).result()
        with self.assertRaises(ValueError):
            est.run([qc2], [op], [[]]).result()
        with self.assertRaises(ValueError):
            est.run([qc], [op], [[1e4]]).result()
        with self.assertRaises(ValueError):
            est.run([qc2], [op2], [[1, 2]]).result()
        with self.assertRaises(ValueError):
            est.run([qc, qc2], [op2], [[1]]).result()
        with self.assertRaises(ValueError):
            est.run([qc], [op, op2], [[1]]).result()

    @combine(backend=BACKENDS)
    def test_run_numpy_params(self, backend):
        """Test for numpy array as parameter values"""
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = BackendEstimator(backend=backend)
        target = estimator.run([qc] * k, [op] * k, params_list).result()

        with self.subTest("ndarrary"):
            result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

        with self.subTest("list of ndarray"):
            result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

    @combine(backend=BACKENDS)
    def test_run_with_shots_option(self, backend):
        """test with shots option."""
        est = BackendEstimator(backend=backend)
        result = est.run(
            [self.ansatz],
            [self.observable],
            parameter_values=[[0, 1, 1, 2, 3, 5]],
            shots=1024,
            seed_simulator=15,
        ).result()
        self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_options(self, backend):
        """Test for options"""
        with self.subTest("init"):
            estimator = BackendEstimator(backend=backend, options={"shots": 3000})
            self.assertEqual(estimator.options.get("shots"), 3000)
        with self.subTest("set_options"):
            estimator.set_options(shots=1024, seed_simulator=15)
            self.assertEqual(estimator.options.get("shots"), 1024)
            self.assertEqual(estimator.options.get("seed_simulator"), 15)
        with self.subTest("run"):
            result = estimator.run(
                [self.ansatz],
                [self.observable],
                parameter_values=[[0, 1, 1, 2, 3, 5]],
            ).result()
            self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    def test_job_size_limit_v2(self):
        """Test BackendEstimator respects job size limit"""

        class FakeNairobiLimitedCircuits(FakeNairobiV2):
            """FakeNairobiV2 with job size limit."""

            @property
            def max_circuits(self):
                return 1

        backend = FakeNairobiLimitedCircuits()
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        reps = 5
        params_array = np.random.rand(reps, qc.num_parameters)
        params_list = params_array.tolist()
        estimator = BackendEstimator(backend=backend)
        obs = len(estimator._observable_decomposer.decompose(op))
        with patch.object(backend, "run") as run_mock:
            estimator.run([qc] * reps, [op] * reps, params_list).result()
        self.assertEqual(run_mock.call_count, reps * obs)

    def test_job_size_limit_v1(self):
        """Test BackendEstimator respects job size limit"""
        backend = FakeNairobi()
        config = backend.configuration()
        config.max_experiments = 1
        backend._configuration = config
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        reps = 5
        params_array = np.random.rand(reps, qc.num_parameters)
        params_list = params_array.tolist()
        estimator = BackendEstimator(backend=backend)
        obs = len(estimator._observable_decomposer.decompose(op))
        with patch.object(backend, "run") as run_mock:
            estimator.run([qc] * reps, [op] * reps, params_list).result()
        self.assertEqual(run_mock.call_count, reps * obs)

    def test_no_max_circuits(self):
        """Test BackendEstimator works with BackendV1 and no max_experiments set."""
        backend = FakeNairobi()
        config = backend.configuration()
        del config.max_experiments
        backend._configuration = config
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = BackendEstimator(backend=backend)
        target = estimator.run([qc] * k, [op] * k, params_list).result()
        with self.subTest("ndarrary"):
            result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

        with self.subTest("list of ndarray"):
            result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)
