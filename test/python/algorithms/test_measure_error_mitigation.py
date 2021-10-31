# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Measurement Error Mitigation """

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data
import numpy as np
import retworkx as rx
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.exceptions import QiskitError
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE, QAOA
from qiskit.opflow import I, X, Z, PauliSumOp
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.utils.mitigation import CompleteMeasFitter, TensoredMeasFitter

try:
    from qiskit import Aer
    from qiskit.providers.aer import noise

    HAS_AER = True
except ImportError:
    HAS_AER = False

try:
    from qiskit.ignis.mitigation.measurement import (
        CompleteMeasFitter as CompleteMeasFitter_IG,
        TensoredMeasFitter as TensoredMeasFitter_IG,
    )

    HAS_IGNIS = True
except ImportError:
    HAS_IGNIS = False


@ddt
class TestMeasurementErrorMitigation(QiskitAlgorithmsTestCase):
    """Test measurement error mitigation."""

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    @data("CompleteMeasFitter", "TensoredMeasFitter")
    def test_measurement_error_mitigation_with_diff_qubit_order(self, fitter_str):
        """measurement error mitigation with different qubit order"""
        algorithm_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        fitter_cls = (
            CompleteMeasFitter if fitter_str == "CompleteMeasFitter" else TensoredMeasFitter
        )
        backend = Aer.get_backend("aer_simulator")
        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=1679,
            seed_transpiler=167,
            shots=1000,
            noise_model=noise_model,
            measurement_error_mitigation_cls=fitter_cls,
            cals_matrix_refresh_period=0,
        )
        # circuit
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure(0, 0)
        qc1.measure(1, 1)
        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.measure(1, 0)
        qc2.measure(0, 1)

        if fitter_cls == TensoredMeasFitter:
            self.assertRaisesRegex(
                QiskitError,
                "TensoredMeasFitter doesn't support subset_fitter.",
                quantum_instance.execute,
                [qc1, qc2],
            )
        else:
            # this should run smoothly
            quantum_instance.execute([qc1, qc2])

        self.assertGreater(quantum_instance.time_taken, 0.0)
        quantum_instance.reset_execution_results()

        # failure case
        qc3 = QuantumCircuit(3, 3)
        qc3.h(2)
        qc3.cx(1, 2)
        qc3.measure(2, 1)
        qc3.measure(1, 2)

        self.assertRaises(QiskitError, quantum_instance.execute, [qc1, qc3])

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    @data(("CompleteMeasFitter", None), ("TensoredMeasFitter", [[0], [1]]))
    def test_measurement_error_mitigation_with_vqe(self, config):
        """measurement error mitigation test with vqe"""

        fitter_str, mit_pattern = config
        algorithm_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        fitter_cls = (
            CompleteMeasFitter if fitter_str == "CompleteMeasFitter" else TensoredMeasFitter
        )
        backend = Aer.get_backend("aer_simulator")
        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=167,
            seed_transpiler=167,
            noise_model=noise_model,
            measurement_error_mitigation_cls=fitter_cls,
            mit_pattern=mit_pattern,
        )

        h2_hamiltonian = (
            -1.052373245772859 * (I ^ I)
            + 0.39793742484318045 * (I ^ Z)
            - 0.39793742484318045 * (Z ^ I)
            - 0.01128010425623538 * (Z ^ Z)
            + 0.18093119978423156 * (X ^ X)
        )
        optimizer = SPSA(maxiter=200)
        ansatz = EfficientSU2(2, reps=1)

        vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
        result = vqe.compute_minimum_eigenvalue(operator=h2_hamiltonian)
        self.assertGreater(quantum_instance.time_taken, 0.0)
        quantum_instance.reset_execution_results()
        self.assertAlmostEqual(result.eigenvalue.real, -1.86, delta=0.05)

    def _get_operator(self, weight_matrix):
        """Generate Hamiltonian for the max-cut problem of a graph.

        Args:
            weight_matrix (numpy.ndarray) : adjacency matrix.

        Returns:
            PauliSumOp: operator for the Hamiltonian
            float: a constant shift for the obj function.

        """
        num_nodes = weight_matrix.shape[0]
        pauli_list = []
        shift = 0
        for i in range(num_nodes):
            for j in range(i):
                if weight_matrix[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([0.5 * weight_matrix[i, j], Pauli((z_p, x_p))])
                    shift -= 0.5 * weight_matrix[i, j]
        opflow_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
        return PauliSumOp.from_list(opflow_list), shift

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    def test_measurement_error_mitigation_qaoa(self):
        """measurement error mitigation test with QAOA"""
        algorithm_globals.random_seed = 167
        backend = Aer.get_backend("aer_simulator")
        w = rx.adjacency_matrix(
            rx.undirected_gnp_random_graph(5, 0.5, seed=algorithm_globals.random_seed)
        )
        qubit_op, _ = self._get_operator(w)
        initial_point = np.asarray([0.0, 0.0])

        # Compute first without noise
        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        qaoa = QAOA(
            optimizer=COBYLA(maxiter=3),
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        ref_eigenvalue = result.eigenvalue.real

        # compute with noise
        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
            noise_model=noise_model,
            measurement_error_mitigation_cls=CompleteMeasFitter,
        )

        qaoa = QAOA(
            optimizer=COBYLA(maxiter=3),
            quantum_instance=quantum_instance,
            initial_point=initial_point,
        )
        result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, ref_eigenvalue, delta=0.05)

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    @unittest.skipUnless(HAS_IGNIS, "qiskit-ignis is required to run this test")
    @data("CompleteMeasFitter", "TensoredMeasFitter")
    def test_measurement_error_mitigation_with_diff_qubit_order_ignis(self, fitter_str):
        """measurement error mitigation with different qubit order"""
        algorithm_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        fitter_cls = (
            CompleteMeasFitter_IG if fitter_str == "CompleteMeasFitter" else TensoredMeasFitter_IG
        )
        backend = Aer.get_backend("aer_simulator")
        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=1679,
            seed_transpiler=167,
            shots=1000,
            noise_model=noise_model,
            measurement_error_mitigation_cls=fitter_cls,
            cals_matrix_refresh_period=0,
        )
        # circuit
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure(0, 0)
        qc1.measure(1, 1)
        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.measure(1, 0)
        qc2.measure(0, 1)

        if fitter_cls == TensoredMeasFitter_IG:
            with self.assertWarnsRegex(DeprecationWarning, r".*ignis.*"):
                self.assertRaisesRegex(
                    QiskitError,
                    "TensoredMeasFitter doesn't support subset_fitter.",
                    quantum_instance.execute,
                    [qc1, qc2],
                )
        else:
            # this should run smoothly
            with self.assertWarnsRegex(DeprecationWarning, r".*ignis.*"):
                quantum_instance.execute([qc1, qc2])

        self.assertGreater(quantum_instance.time_taken, 0.0)
        quantum_instance.reset_execution_results()

        # failure case
        qc3 = QuantumCircuit(3, 3)
        qc3.h(2)
        qc3.cx(1, 2)
        qc3.measure(2, 1)
        qc3.measure(1, 2)

        self.assertRaises(QiskitError, quantum_instance.execute, [qc1, qc3])

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    @unittest.skipUnless(HAS_IGNIS, "qiskit-ignis is required to run this test")
    @data(("CompleteMeasFitter", None), ("TensoredMeasFitter", [[0], [1]]))
    def test_measurement_error_mitigation_with_vqe_ignis(self, config):
        """measurement error mitigation test with vqe"""
        fitter_str, mit_pattern = config
        algorithm_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        fitter_cls = (
            CompleteMeasFitter_IG if fitter_str == "CompleteMeasFitter" else TensoredMeasFitter_IG
        )
        backend = Aer.get_backend("aer_simulator")
        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=167,
            seed_transpiler=167,
            noise_model=noise_model,
            measurement_error_mitigation_cls=fitter_cls,
            mit_pattern=mit_pattern,
        )

        h2_hamiltonian = (
            -1.052373245772859 * (I ^ I)
            + 0.39793742484318045 * (I ^ Z)
            - 0.39793742484318045 * (Z ^ I)
            - 0.01128010425623538 * (Z ^ Z)
            + 0.18093119978423156 * (X ^ X)
        )
        optimizer = SPSA(maxiter=200)
        ansatz = EfficientSU2(2, reps=1)

        vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)
        with self.assertWarnsRegex(DeprecationWarning, r".*ignis.*"):
            result = vqe.compute_minimum_eigenvalue(operator=h2_hamiltonian)
        self.assertGreater(quantum_instance.time_taken, 0.0)
        quantum_instance.reset_execution_results()
        self.assertAlmostEqual(result.eigenvalue.real, -1.86, delta=0.05)

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    @unittest.skipUnless(HAS_IGNIS, "qiskit-ignis is required to run this test")
    def test_callibration_results(self):
        """check that results counts are the same with/without error mitigation"""
        algorithm_globals.random_seed = 1679
        np.random.seed(algorithm_globals.random_seed)
        qc = QuantumCircuit(1)
        qc.x(0)

        qc_meas = qc.copy()
        qc_meas.measure_all()
        backend = Aer.get_backend("aer_simulator")

        counts_array = [None, None]
        for idx, is_use_mitigation in enumerate([True, False]):
            if is_use_mitigation:
                quantum_instance = QuantumInstance(
                    backend,
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                    shots=1024,
                    measurement_error_mitigation_cls=CompleteMeasFitter_IG,
                )
                with self.assertWarnsRegex(DeprecationWarning, r".*ignis.*"):
                    counts_array[idx] = quantum_instance.execute(qc_meas).get_counts()
            else:
                quantum_instance = QuantumInstance(
                    backend,
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                    shots=1024,
                )
                counts_array[idx] = quantum_instance.execute(qc_meas).get_counts()
        self.assertEqual(
            counts_array[0], counts_array[1], msg="Counts different with/without fitter."
        )


if __name__ == "__main__":
    unittest.main()
