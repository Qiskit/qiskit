# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Test of measurement calibration:
1) Preparation of the basis states, generating the calibration circuits
(without noise), computing the calibration matrices,
and validating that they equal
to the identity matrices
2) Generating ideal (equally distributed) results, computing
the calibration output (without noise),
and validating that it is equally distributed
3) Testing the the measurement calibration on a circuit
(without noise), verifying that it is close to the
expected (equally distributed) result
4) Testing the fitters on pre-generated data with noise
"""

import unittest
import numpy as np

import qiskit
from qiskit.test import QiskitTestCase
from qiskit.result.result import Result
from qiskit.utils.mitigation import (
    CompleteMeasFitter,
    TensoredMeasFitter,
    complete_meas_cal,
    tensored_meas_cal,
)
from qiskit.utils.mitigation._filters import MeasurementFilter
from qiskit.utils.mitigation.circuits import count_keys

from qiskit.utils import optionals

if optionals.HAS_AER:
    # pylint: disable=no-name-in-module
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.errors.standard_errors import pauli_error

# fixed seed for tests - for both simulator and transpiler
SEED = 42


def convert_ndarray_to_list_in_data(data: np.ndarray):
    """
    converts ndarray format into list format (keeps all the dicts in the array)
    also convert inner ndarrays into lists (recursively)
    Args:
        data: ndarray containing dicts or ndarrays in it

    Returns:
        list: same array, converted to list format (in order to save it as json)

    """
    new_data = []
    for item in data:
        if isinstance(item, np.ndarray):
            new_item = convert_ndarray_to_list_in_data(item)
        elif isinstance(item, dict):
            new_item = {}
            for key, value in item.items():
                new_item[key] = value.tolist()
        else:
            new_item = item
        new_data.append(new_item)

    return new_data


def meas_calib_circ_creation():
    """
    create measurement calibration circuits and a GHZ state circuit for the tests

    Returns:
        QuantumCircuit: the measurement calibrations circuits
        list[str]: the mitigation pattern
        QuantumCircuit: ghz circuit with 5 qubits (3 are used)

    """
    qubit_list = [1, 2, 3]
    total_number_of_qubit = 5
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=total_number_of_qubit)

    # Choose 3 qubits
    qubit_1 = qubit_list[0]
    qubit_2 = qubit_list[1]
    qubit_3 = qubit_list[2]
    ghz = qiskit.QuantumCircuit(total_number_of_qubit, len(qubit_list))
    ghz.h(qubit_1)
    ghz.cx(qubit_1, qubit_2)
    ghz.cx(qubit_1, qubit_3)
    for i in qubit_list:
        ghz.measure(i, i - 1)
    return meas_calibs, state_labels, ghz


def tensored_calib_circ_creation():
    """
    create tensored measurement calibration circuits and a GHZ state circuit for the tests

    Returns:
        QuantumCircuit: the tensored measurement calibration circuit
        list[list[int]]: the mitigation pattern
        QuantumCircuit: ghz circuit with 5 qubits (3 are used)

    """
    mit_pattern = [[2], [4, 1]]
    meas_layout = [2, 4, 1]
    qr = qiskit.QuantumRegister(5)
    # Generate the calibration circuits
    meas_calibs, mit_pattern = tensored_meas_cal(mit_pattern, qr=qr)

    cr = qiskit.ClassicalRegister(3)
    ghz_circ = qiskit.QuantumCircuit(qr, cr)
    ghz_circ.h(mit_pattern[0][0])
    ghz_circ.cx(mit_pattern[0][0], mit_pattern[1][0])
    ghz_circ.cx(mit_pattern[0][0], mit_pattern[1][1])
    ghz_circ.measure(mit_pattern[0][0], cr[0])
    ghz_circ.measure(mit_pattern[1][0], cr[1])
    ghz_circ.measure(mit_pattern[1][1], cr[2])
    return meas_calibs, mit_pattern, ghz_circ, meas_layout


def meas_calibration_circ_execution(shots: int, seed: int):
    """
    create measurement calibration circuits and simulate them with noise
    Args:
        shots (int): number of shots per simulation
        seed (int): the seed to use in the simulations

    Returns:
        list: list of Results of the measurement calibration simulations
        list: list of all the possible states with this amount of qubits
        dict: dictionary of results counts of GHZ circuit simulation with measurement errors
    """
    # define the circuits
    meas_calibs, state_labels, ghz = meas_calib_circ_creation()

    # define noise
    prob = 0.2
    error_meas = pauli_error([("X", prob), ("I", 1 - prob)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    # run the circuits multiple times
    backend = AerSimulator()
    cal_results = qiskit.execute(
        meas_calibs, backend=backend, shots=shots, noise_model=noise_model, seed_simulator=seed
    ).result()

    ghz_results = (
        qiskit.execute(
            ghz, backend=backend, shots=shots, noise_model=noise_model, seed_simulator=seed
        )
        .result()
        .get_counts()
    )

    return cal_results, state_labels, ghz_results


def tensored_calib_circ_execution(shots: int, seed: int):
    """
    create tensored measurement calibration circuits and simulate them with noise
    Args:
        shots (int): number of shots per simulation
        seed (int): the seed to use in the simulations

    Returns:
        list: list of Results of the measurement calibration simulations
        list: the mitigation pattern
        dict: dictionary of results counts of GHZ circuit simulation with measurement errors
    """
    # define the circuits
    meas_calibs, mit_pattern, ghz_circ, meas_layout = tensored_calib_circ_creation()
    # define noise
    prob = 0.2
    error_meas = pauli_error([("X", prob), ("I", 1 - prob)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    # run the circuits multiple times
    backend = AerSimulator()
    cal_results = qiskit.execute(
        meas_calibs, backend=backend, shots=shots, noise_model=noise_model, seed_simulator=seed
    ).result()

    ghz_results = qiskit.execute(
        ghz_circ, backend=backend, shots=shots, noise_model=noise_model, seed_simulator=seed
    ).result()

    return cal_results, mit_pattern, ghz_results, meas_layout


@unittest.skipUnless(optionals.HAS_AER, "Qiskit aer is required to run these tests")
class TestMeasCal(QiskitTestCase):
    """The test class."""

    def setUp(self):
        super().setUp()
        self.nq_list = [1, 2, 3, 4, 5]  # Test up to 5 qubits
        self.shots = 1024  # Number of shots (should be a power of 2)

    @staticmethod
    def choose_calibration(nq, pattern_type):
        """
        Generate a calibration circuit

        Args:
            nq (int): number of qubits
            pattern_type (int): a pattern in range(1, 2**nq)

        Returns:
            qubits: a list of qubits according to the given pattern
            weight: the weight of the pattern_type,
                    equals to the number of qubits

        Additional Information:
            qr[i] exists if and only if the i-th bit in the binary
            expression of
            pattern_type equals 1
        """
        qubits = []
        weight = 0
        for i in range(nq):
            pattern_bit = pattern_type & 1
            pattern_type = pattern_type >> 1
            if pattern_bit == 1:
                qubits.append(i)
                weight += 1
        return qubits, weight

    def generate_ideal_results(self, state_labels, weight):
        """
        Generate ideal equally distributed results

        Args:
            state_labels (list): a list of calibration state labels
            weight (int): the number of qubits

        Returns:
            results_dict: a dictionary of equally distributed results
            results_list: a list of equally distributed results

        Additional Information:
            for each state in state_labels:
            result_dict[state] = #shots/len(state_labels)
        """
        results_dict = {}
        results_list = [0] * (2**weight)
        state_num = len(state_labels)
        for state in state_labels:
            shots_per_state = self.shots / state_num
            results_dict[state] = shots_per_state
            # converting state (binary) to an integer
            place = int(state, 2)
            results_list[place] = shots_per_state
        return results_dict, results_list

    def test_ideal_meas_cal(self):
        """Test ideal execution, without noise."""
        for nq in self.nq_list:
            for pattern_type in range(1, 2**nq):

                # Generate the quantum register according to the pattern
                qubits, weight = self.choose_calibration(nq, pattern_type)

                with self.assertWarns(DeprecationWarning):
                    # Generate the calibration circuits
                    meas_calibs, state_labels = complete_meas_cal(
                        qubit_list=qubits, circlabel="test"
                    )

                # Perform an ideal execution on the generated circuits
                backend = AerSimulator()
                job = qiskit.execute(meas_calibs, backend=backend, shots=self.shots)
                cal_results = job.result()

                with self.assertWarns(DeprecationWarning):
                    # Make a calibration matrix
                    meas_cal = CompleteMeasFitter(cal_results, state_labels, circlabel="test")

                # Assert that the calibration matrix is equal to identity
                IdentityMatrix = np.identity(2**weight)
                self.assertListEqual(
                    meas_cal.cal_matrix.tolist(),
                    IdentityMatrix.tolist(),
                    "Error: the calibration matrix is not equal to identity",
                )

                # Assert that the readout fidelity is equal to 1
                self.assertEqual(
                    meas_cal.readout_fidelity(),
                    1.0,
                    "Error: the average fidelity is not equal to 1",
                )

                # Generate ideal (equally distributed) results
                results_dict, results_list = self.generate_ideal_results(state_labels, weight)

                with self.assertWarns(DeprecationWarning):
                    # Output the filter
                    meas_filter = meas_cal.filter

                # Apply the calibration matrix to results
                # in list and dict forms using different methods
                results_dict_1 = meas_filter.apply(results_dict, method="least_squares")
                results_dict_0 = meas_filter.apply(results_dict, method="pseudo_inverse")
                results_list_1 = meas_filter.apply(results_list, method="least_squares")
                results_list_0 = meas_filter.apply(results_list, method="pseudo_inverse")

                # Assert that the results are equally distributed
                self.assertListEqual(results_list, results_list_0.tolist())
                self.assertListEqual(results_list, np.round(results_list_1).tolist())
                self.assertDictEqual(results_dict, results_dict_0)
                round_results = {}
                for key, val in results_dict_1.items():
                    round_results[key] = np.round(val)
                self.assertDictEqual(results_dict, round_results)

    def test_meas_cal_on_circuit(self):
        """Test an execution on a circuit."""
        # Generate the calibration circuits
        with self.assertWarns(DeprecationWarning):
            meas_calibs, state_labels, ghz = meas_calib_circ_creation()

        # Run the calibration circuits
        backend = AerSimulator()
        job = qiskit.execute(
            meas_calibs,
            backend=backend,
            shots=self.shots,
            seed_simulator=SEED,
            seed_transpiler=SEED,
        )
        cal_results = job.result()

        with self.assertWarns(DeprecationWarning):
            # Make a calibration matrix
            meas_cal = CompleteMeasFitter(cal_results, state_labels)
        # Calculate the fidelity
        fidelity = meas_cal.readout_fidelity()

        job = qiskit.execute(
            [ghz], backend=backend, shots=self.shots, seed_simulator=SEED, seed_transpiler=SEED
        )
        results = job.result()

        # Predicted equally distributed results
        predicted_results = {"000": 0.5, "111": 0.5}

        with self.assertWarns(DeprecationWarning):
            meas_filter = meas_cal.filter

        # Calculate the results after mitigation
        output_results_pseudo_inverse = meas_filter.apply(
            results, method="pseudo_inverse"
        ).get_counts(0)
        output_results_least_square = meas_filter.apply(results, method="least_squares").get_counts(
            0
        )

        # Compare with expected fidelity and expected results
        self.assertAlmostEqual(fidelity, 1.0)
        self.assertAlmostEqual(
            output_results_pseudo_inverse["000"] / self.shots, predicted_results["000"], places=1
        )

        self.assertAlmostEqual(
            output_results_least_square["000"] / self.shots, predicted_results["000"], places=1
        )

        self.assertAlmostEqual(
            output_results_pseudo_inverse["111"] / self.shots, predicted_results["111"], places=1
        )

        self.assertAlmostEqual(
            output_results_least_square["111"] / self.shots, predicted_results["111"], places=1
        )

    def test_ideal_tensored_meas_cal(self):
        """Test ideal execution, without noise."""

        mit_pattern = [[1, 2], [3, 4, 5], [6]]
        meas_layout = [1, 2, 3, 4, 5, 6]

        # Generate the calibration circuits
        with self.assertWarns(DeprecationWarning):
            meas_calibs, _ = tensored_meas_cal(mit_pattern=mit_pattern)

        # Perform an ideal execution on the generated circuits
        backend = AerSimulator()
        cal_results = qiskit.execute(meas_calibs, backend=backend, shots=self.shots).result()

        with self.assertWarns(DeprecationWarning):
            # Make calibration matrices
            meas_cal = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)

        # Assert that the calibration matrices are equal to identity
        cal_matrices = meas_cal.cal_matrices
        self.assertEqual(
            len(mit_pattern), len(cal_matrices), "Wrong number of calibration matrices"
        )
        for qubit_list, cal_mat in zip(mit_pattern, cal_matrices):
            IdentityMatrix = np.identity(2 ** len(qubit_list))
            self.assertListEqual(
                cal_mat.tolist(),
                IdentityMatrix.tolist(),
                "Error: the calibration matrix is not equal to identity",
            )

        # Assert that the readout fidelity is equal to 1
        self.assertEqual(
            meas_cal.readout_fidelity(),
            1.0,
            "Error: the average fidelity is not equal to 1",
        )

        with self.assertWarns(DeprecationWarning):
            # Generate ideal (equally distributed) results
            results_dict, _ = self.generate_ideal_results(count_keys(6), 6)
            # Output the filter
            meas_filter = meas_cal.filter
            # Apply the calibration matrix to results
            # in list and dict forms using different methods
            results_dict_1 = meas_filter.apply(
                results_dict, method="least_squares", meas_layout=meas_layout
            )
            results_dict_0 = meas_filter.apply(
                results_dict, method="pseudo_inverse", meas_layout=meas_layout
            )

        # Assert that the results are equally distributed
        self.assertDictEqual(results_dict, results_dict_0)
        round_results = {}
        for key, val in results_dict_1.items():
            round_results[key] = np.round(val)
        self.assertDictEqual(results_dict, round_results)

    def test_tensored_meas_cal_on_circuit(self):
        """Test an execution on a circuit."""

        with self.assertWarns(DeprecationWarning):
            # Generate the calibration circuits
            meas_calibs, mit_pattern, ghz, meas_layout = tensored_calib_circ_creation()

        # Run the calibration circuits
        backend = AerSimulator()
        cal_results = qiskit.execute(
            meas_calibs,
            backend=backend,
            shots=self.shots,
            seed_simulator=SEED,
            seed_transpiler=SEED,
        ).result()

        with self.assertWarns(DeprecationWarning):
            # Make a calibration matrix
            meas_cal = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
        # Calculate the fidelity
        fidelity = meas_cal.readout_fidelity(0) * meas_cal.readout_fidelity(1)

        results = qiskit.execute(
            [ghz], backend=backend, shots=self.shots, seed_simulator=SEED, seed_transpiler=SEED
        ).result()

        # Predicted equally distributed results
        predicted_results = {"000": 0.5, "111": 0.5}

        with self.assertWarns(DeprecationWarning):
            meas_filter = meas_cal.filter
            # Calculate the results after mitigation
            output_results_pseudo_inverse = meas_filter.apply(
                results, method="pseudo_inverse", meas_layout=meas_layout
            ).get_counts(0)
            output_results_least_square = meas_filter.apply(
                results, method="least_squares", meas_layout=meas_layout
            ).get_counts(0)

        # Compare with expected fidelity and expected results
        self.assertAlmostEqual(fidelity, 1.0)
        self.assertAlmostEqual(
            output_results_pseudo_inverse["000"] / self.shots, predicted_results["000"], places=1
        )

        self.assertAlmostEqual(
            output_results_least_square["000"] / self.shots, predicted_results["000"], places=1
        )

        self.assertAlmostEqual(
            output_results_pseudo_inverse["111"] / self.shots, predicted_results["111"], places=1
        )

        self.assertAlmostEqual(
            output_results_least_square["111"] / self.shots, predicted_results["111"], places=1
        )

    def test_meas_fitter_with_noise(self):
        """Test the MeasurementFitter with noise."""
        tests = []
        runs = 3
        with self.assertWarns(DeprecationWarning):
            for run in range(runs):
                cal_results, state_labels, circuit_results = meas_calibration_circ_execution(
                    1000, SEED + run
                )

                meas_cal = CompleteMeasFitter(cal_results, state_labels)
                meas_filter = MeasurementFilter(meas_cal.cal_matrix, state_labels)

                # Calculate the results after mitigation
                results_pseudo_inverse = meas_filter.apply(circuit_results, method="pseudo_inverse")
                results_least_square = meas_filter.apply(circuit_results, method="least_squares")
                tests.append(
                    {
                        "cal_matrix": convert_ndarray_to_list_in_data(meas_cal.cal_matrix),
                        "fidelity": meas_cal.readout_fidelity(),
                        "results": circuit_results,
                        "results_pseudo_inverse": results_pseudo_inverse,
                        "results_least_square": results_least_square,
                    }
                )
            # Set the state labels
            state_labels = ["000", "001", "010", "011", "100", "101", "110", "111"]
            meas_cal = CompleteMeasFitter(None, state_labels, circlabel="test")

            for tst_index, _ in enumerate(tests):
                # Set the calibration matrix
                meas_cal.cal_matrix = tests[tst_index]["cal_matrix"]
                # Calculate the fidelity
                fidelity = meas_cal.readout_fidelity()

                meas_filter = MeasurementFilter(tests[tst_index]["cal_matrix"], state_labels)

                # Calculate the results after mitigation
                output_results_pseudo_inverse = meas_filter.apply(
                    tests[tst_index]["results"], method="pseudo_inverse"
                )
                output_results_least_square = meas_filter.apply(
                    tests[tst_index]["results"], method="least_squares"
                )

                # Compare with expected fidelity and expected results
                self.assertAlmostEqual(fidelity, tests[tst_index]["fidelity"], places=0)
                self.assertAlmostEqual(
                    output_results_pseudo_inverse["000"],
                    tests[tst_index]["results_pseudo_inverse"]["000"],
                    places=0,
                )

                self.assertAlmostEqual(
                    output_results_least_square["000"],
                    tests[tst_index]["results_least_square"]["000"],
                    places=0,
                )

                self.assertAlmostEqual(
                    output_results_pseudo_inverse["111"],
                    tests[tst_index]["results_pseudo_inverse"]["111"],
                    places=0,
                )

                self.assertAlmostEqual(
                    output_results_least_square["111"],
                    tests[tst_index]["results_least_square"]["111"],
                    places=0,
                )

    def test_tensored_meas_fitter_with_noise(self):
        """Test the TensoredFitter with noise."""
        with self.assertWarns(DeprecationWarning):
            cal_results, mit_pattern, circuit_results, meas_layout = tensored_calib_circ_execution(
                1000, SEED
            )
            meas_cal = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
            meas_filter = meas_cal.filter
            # Calculate the results after mitigation
            results_pseudo_inverse = meas_filter.apply(
                circuit_results.get_counts(), method="pseudo_inverse", meas_layout=meas_layout
            )
            results_least_square = meas_filter.apply(
                circuit_results.get_counts(), method="least_squares", meas_layout=meas_layout
            )

        saved_info = {
            "cal_results": cal_results.to_dict(),
            "results": circuit_results.to_dict(),
            "mit_pattern": mit_pattern,
            "meas_layout": meas_layout,
            "fidelity": meas_cal.readout_fidelity(),
            "results_pseudo_inverse": results_pseudo_inverse,
            "results_least_square": results_least_square,
        }

        saved_info["cal_results"] = Result.from_dict(saved_info["cal_results"])
        saved_info["results"] = Result.from_dict(saved_info["results"])

        with self.assertWarns(DeprecationWarning):
            meas_cal = TensoredMeasFitter(
                saved_info["cal_results"], mit_pattern=saved_info["mit_pattern"]
            )
            # Calculate the fidelity
            fidelity = meas_cal.readout_fidelity(0) * meas_cal.readout_fidelity(1)
            # Compare with expected fidelity and expected results

        self.assertAlmostEqual(fidelity, saved_info["fidelity"], places=0)

        with self.assertWarns(DeprecationWarning):
            meas_filter = meas_cal.filter
            # Calculate the results after mitigation
            output_results_pseudo_inverse = meas_filter.apply(
                saved_info["results"].get_counts(0),
                method="pseudo_inverse",
                meas_layout=saved_info["meas_layout"],
            )
            output_results_least_square = meas_filter.apply(
                saved_info["results"], method="least_squares", meas_layout=saved_info["meas_layout"]
            )

        self.assertAlmostEqual(
            output_results_pseudo_inverse["000"],
            saved_info["results_pseudo_inverse"]["000"],
            places=0,
        )

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)["000"],
            saved_info["results_least_square"]["000"],
            places=0,
        )

        self.assertAlmostEqual(
            output_results_pseudo_inverse["111"],
            saved_info["results_pseudo_inverse"]["111"],
            places=0,
        )

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)["111"],
            saved_info["results_least_square"]["111"],
            places=0,
        )

        substates_list = []
        with self.assertWarns(DeprecationWarning):
            for qubit_list in saved_info["mit_pattern"]:
                substates_list.append(count_keys(len(qubit_list))[::-1])
            fitter_other_order = TensoredMeasFitter(
                saved_info["cal_results"],
                substate_labels_list=substates_list,
                mit_pattern=saved_info["mit_pattern"],
            )

        fidelity = fitter_other_order.readout_fidelity(0) * meas_cal.readout_fidelity(1)

        self.assertAlmostEqual(fidelity, saved_info["fidelity"], places=0)

        with self.assertWarns(DeprecationWarning):
            meas_filter = fitter_other_order.filter
            # Calculate the results after mitigation
            output_results_pseudo_inverse = meas_filter.apply(
                saved_info["results"].get_counts(0),
                method="pseudo_inverse",
                meas_layout=saved_info["meas_layout"],
            )
            output_results_least_square = meas_filter.apply(
                saved_info["results"], method="least_squares", meas_layout=saved_info["meas_layout"]
            )

        self.assertAlmostEqual(
            output_results_pseudo_inverse["000"],
            saved_info["results_pseudo_inverse"]["000"],
            places=0,
        )

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)["000"],
            saved_info["results_least_square"]["000"],
            places=0,
        )

        self.assertAlmostEqual(
            output_results_pseudo_inverse["111"],
            saved_info["results_pseudo_inverse"]["111"],
            places=0,
        )

        self.assertAlmostEqual(
            output_results_least_square.get_counts(0)["111"],
            saved_info["results_least_square"]["111"],
            places=0,
        )


if __name__ == "__main__":
    unittest.main()
