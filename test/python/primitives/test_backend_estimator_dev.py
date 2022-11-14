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
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from ddt import ddt, data, unpack
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives.backend_estimator_dev import (
    AbelianDecomposer,
    BackendEstimator,
    NaiveDecomposer,
)
from qiskit.providers import Backend, Options
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import ApplyLayout, SetLayout


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


def circuit_composition_examples() -> Iterator[
    tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]
]:
    """Generator of base and measurement circuits, and respective composition.

    Yields:
        - Transpiled base circuit: with a `final_layout` entry in its metadata
        - Measurement circuit: before transpilation (i.e. no layout applied)
        - Transpiled measurement circuit: with `final_layout` applied
    """
    target_qubits = 2
    layout_intlist = (0,)
    measured_qubits = (0,)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 2
    layout_intlist = (1,)
    measured_qubits = (0,)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 2
    layout_intlist = (1, 0)
    measured_qubits = (1,)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 2
    layout_intlist = (1, 0)
    measured_qubits = (0, 1)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 3
    layout_intlist = (2,)
    measured_qubits = (0,)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 3
    layout_intlist = (2, 0)
    measured_qubits = (0, 1)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 3
    layout_intlist = (1, 2)
    measured_qubits = (1,)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 3
    layout_intlist = (1, 0, 2)
    measured_qubits = (1, 2)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 4
    layout_intlist = (0, 1, 2)
    measured_qubits = (0, 1)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 4
    layout_intlist = (1, 0, 3)
    measured_qubits = (1, 2)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 4
    layout_intlist = (2, 1, 3, 0)
    measured_qubits = (0, 1, 2)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)

    target_qubits = 4
    layout_intlist = (0, 2, 1, 3)
    measured_qubits = (0, 1, 2, 3)
    yield build_composition_data(target_qubits, layout_intlist, measured_qubits)


def build_composition_data(
    target_qubits: int, layout_intlist: Sequence[int], measured_qubits: Sequence[int]
) -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Build base and measurement circuits, and respective composition.

    Args:
        - target_qubits: the number of qubits to target during transpilation
        - layout_intlist: indices to map virtual qubits to during transpilaiton
        - measured_qubits: virtual qubits to measure

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
class TestTranspilation(TestCase):
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
        backend = Mock(Backend)
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
        backend = Mock(Backend)
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
class TestMeasurement(TestCase):
    """Test measurement logic."""

    def test_observable_decomposer(self):
        """Test observable decomposer property."""
        estimator = BackendEstimator(Mock(Backend))
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
        circuit = BackendEstimator(Mock(Backend))._build_single_measurement_circuit(observable)
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit, measurement)
        self.assertIsInstance(circuit.metadata.get("measured_qubit_indices"), tuple)
        self.assertEqual(circuit.metadata.get("measured_qubit_indices"), meas_indices)
        self.assertIsInstance(circuit.metadata.get("paulis"), PauliList)
        self.assertEqual(circuit.metadata.get("paulis"), paulis)
        self.assertIsInstance(circuit.metadata.get("coeffs"), tuple)
        self.assertEqual(circuit.metadata.get("coeffs"), coeffs)


@ddt
class TestComposition(TestCase):
    """Test composition logic."""

    @data(*circuit_composition_examples())
    @unpack
    def test_compose_single_measurement(self, transpiled_base, measurement, transpiled_measurement):
        """Test coposition of single base circuit and measurement pair."""
        # Preapration
        expected_composition = transpiled_base.compose(transpiled_measurement)
        expected_metadata = {**transpiled_base.metadata, **measurement.metadata}
        expected_metadata.pop("measured_qubit_indices")
        # Test
        backend = Mock(Backend)
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


# TODO
@ddt
class TestCalculations(TestCase):
    """Test calculation logic."""

    @data(
        [("II",), (0,)],
        [("IZ",), (1,)],
        [("ZI",), (2,)],
        [("ZZ",), (3,)],
        [("ZX",), (3,)],
        [("XY",), (3,)],
        [("IIII",), (0,)],
        [("IXII",), (4,)],
        [("IIX", "IYI"), (1, 2)],
        [("IXYZ", "IIII", "IZZZ"), (7, 0, 7)],
    )
    @unpack
    def test_paulis_integer_masks(self, paulis, expected):
        """Test Paulis integer masks."""
        paulis = PauliList(paulis)
        int_masks = BackendEstimator._paulis_integer_masks(paulis)
        self.assertEqual(int_masks, expected)

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
class TestObservableDecomposer(TestCase):
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
