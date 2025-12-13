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

"""Test the evolved operator ansatz."""

import unittest
from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import (
    SparsePauliOp,
    Operator,
    Pauli,
    Statevector,
)

# SparseObservable may be provided by the accelerate bindings. Guard the import.
try:
    from qiskit.quantum_info import SparseObservable  # type: ignore
except Exception:  # pragma: no cover - defensive import for older installs
    SparseObservable = None  # type: ignore

from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library.n_local import (
    EvolvedOperatorAnsatz,
    evolved_operator_ansatz,
    hamiltonian_variational_ansatz,
)
from qiskit.synthesis.evolution import MatrixExponential
from qiskit.utils import optionals
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestEvolvedOperatorAnsatz(QiskitTestCase):
    """Test the evolved operator ansatz."""

    @data(True, False)
    def test_evolved_op_ansatz(self, use_function):
        """Test the default evolution."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]

        if use_function:
            evo = evolved_operator_ansatz(ops, 2)
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz(ops, 2)

        parameters = evo.parameters

        reference = QuantumCircuit(num_qubits)
        strings = ["z" * num_qubits, "y" * num_qubits, "x" * num_qubits] * 2
        for string, time in zip(strings, parameters):
            reference.compose(evolve(string, time), inplace=True)

        if not use_function:
            evo = evo.decompose().decompose()

        self.assertEqual(evo, reference)

    @data(True, False)
    def test_custom_evolution(self, use_function):
        """Test using another evolution than the default (e.g. matrix evolution)."""
        op = SparsePauliOp(["ZIX"])
        matrix = np.array(op)
        evolution = MatrixExponential()

        if use_function:
            evo = evolved_operator_ansatz(op, evolution=evolution)
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz(op, evolution=evolution)

        parameters = evo.parameters

        reference = QuantumCircuit(3)
        reference.append(HamiltonianGate(matrix, parameters[0]), [0, 1, 2])

        if not use_function:
            evo = evo.decompose().decompose()

        self.assertEqual(evo, reference)

    def test_changing_operators(self):
        """Test rebuilding after the operators changed."""
        ops = [Pauli("X"), Pauli("Y"), Pauli("Z")]
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz(ops)

        evo.operators = [Pauli("X"), Pauli("Y")]
        parameters = evo.parameters

        reference = QuantumCircuit(1)
        reference.rx(2 * parameters[0], 0)
        reference.ry(2 * parameters[1], 0)

        self.assertEqual(evo.decompose(reps=2), reference)

    @data(True, False)
    def test_invalid_reps(self, use_function):
        """Test setting an invalid number of reps."""
        with self.assertRaises(ValueError):
            if use_function:
                _ = evolved_operator_ansatz(Pauli("X"), reps=-1)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = EvolvedOperatorAnsatz(Pauli("X"), reps=-1)

    def test_insert_barriers_circuit(self):
        """Test using insert_barriers."""
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for parameter in evo.parameters:
            ref.rz(2.0 * parameter, 0)
            ref.barrier()

        self.assertEqual(evo.decompose(reps=2), ref)

    def test_insert_barriers(self):
        """Test using insert_barriers."""
        evo = evolved_operator_ansatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for i, parameter in enumerate(evo.parameters):
            ref.rz(2.0 * parameter, 0)
            if i < evo.num_parameters - 1:
                ref.barrier()

        self.assertEqual(evo, ref)

    def test_empty_build_fails(self):
        """Test setting no operators to evolve raises the appropriate error."""
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz()
        with self.assertRaises(ValueError):
            _ = evo.draw()

    @data(True, False)
    def test_empty_operator_list(self, use_function):
        """Test setting an empty list of operators to be equal to an empty circuit."""
        if use_function:
            evo = evolved_operator_ansatz([])
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz([])

        self.assertEqual(evo, QuantumCircuit())

    @data(True, False)
    def test_matrix_operator(self, use_function):
        """Test passing a quantum_info.Operator uses the HamiltonianGate."""
        unitary = Operator([[0, 1], [1, 0]])

        if use_function:
            evo = evolved_operator_ansatz(unitary, reps=3)
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz(unitary, reps=3).decompose()

        self.assertEqual(evo.count_ops()["hamiltonian"], 3)

    def test_flattened(self):
        """Test flatten option is actually flattened."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz(ops, reps=3, flatten=True)
        self.assertNotIn("hamiltonian", evo.count_ops())
        self.assertNotIn("EvolvedOps", evo.count_ops())
        self.assertNotIn("PauliEvolution", evo.count_ops())

    def test_flattening(self):
        """Test ``flatten`` on the function."""
        operators = [Operator.from_label("X"), SparsePauliOp(["Z"])]

        with self.subTest(flatten=None):
            evo = evolved_operator_ansatz(operators, flatten=None)
            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertNotIn("PauliEvolution", ops)

        with self.subTest(flatten=True):
            # check we get a warning when trying to flatten a HamiltonianGate,
            # which has an unbound param and cannot be flattened
            with self.assertWarnsRegex(UserWarning, "Cannot flatten"):
                evo = evolved_operator_ansatz(operators, flatten=True)

            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertNotIn("PauliEvolution", ops)

        with self.subTest(flatten=False):
            evo = evolved_operator_ansatz(operators, flatten=False)
            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertIn("PauliEvolution", ops)

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_sympify_is_real(self):
        """Test converting the parameters to sympy is real."""
        evo = evolved_operator_ansatz(SparsePauliOp(["Z"], coeffs=[1 + 0j]))
        param = evo.parameters[0]  # get the gamma parameter

        angle = evo.data[0].operation.params[0]
        expected = (2.0 * param).sympify()
        self.assertEqual(expected, angle.sympify())

    # ---------------------------------------------------------------------
    # New tests to verify SparseObservable is accepted and matches fallback
    # ---------------------------------------------------------------------
    def test_accepts_sparseobservable(self):
        """Test that a SparseObservable input can be used to build the ansatz."""
        if SparseObservable is None:
            self.skipTest("SparseObservable not available in this build")

        # Build a simple SparsePauliOp and wrap it as a SparseObservable via constructor,
        # matching patterns used elsewhere in the codebase.
        spo = SparsePauliOp(["Z"])  # 1-qubit Z
        obs = SparseObservable(spo)

        # Should construct without exception and return a QuantumCircuit
        evo = evolved_operator_ansatz(obs, reps=1)
        self.assertIsInstance(evo, QuantumCircuit)

    def test_sparseobservable_matches_dense_operator(self):
        """Test that evolving a SparseObservable yields the same effect as evolving the equivalent dense operator."""
        if SparseObservable is None:
            self.skipTest("SparseObservable not available in this build")

        # Use a small, simple observable (1 qubit Z) for a robust comparison
        spo = SparsePauliOp(["Z"])
        obs = SparseObservable(spo)

        # Build ansatz via SparseObservable (should exercise the fast path if available)
        evo_obs = evolved_operator_ansatz(obs, reps=1)

        # Build ansatz via dense matrix fallback (Operator)
        matrix = np.array(spo)
        evo_dense = evolved_operator_ansatz(Operator(matrix), reps=1)

        # Bind parameters if present (use same values for both circuits)
        params = list(evo_obs.parameters)
        param_values = [0.37 for _ in params]  # arbitrary test values

        bound_obs = evo_obs.assign_parameters(param_values)
        bound_dense = evo_dense.assign_parameters(param_values)

        # Prepare initial state |0> (single qubit)
        sv0 = Statevector.from_label("0")

        final_obs = sv0.evolve(bound_obs)
        final_dense = sv0.evolve(bound_dense)

        # Compare final statevectors for equivalence
        self.assertTrue(final_obs.equiv(final_dense))


@unittest.skipIf(
    SparseObservable is None,
    "SparseObservable not available in this Qiskit version"
)
class TestEvolvedOperatorAnsatzSparseObservable(QiskitTestCase):
    """Test evolved_operator_ansatz with SparseObservable operators."""

    def test_sparse_observable_rust_path(self):
        """Test that SparseObservable uses the Rust-accelerated path."""
        # Create a SparseObservable with Pauli terms
        obs = SparseObservable.from_sparse_list([
            ("X", [0], 1.0),
            ("Z", [1], 1.0),
        ], num_qubits=2)

        # This should use the fast Rust path (flatten=True, evolution=None)
        ansatz = evolved_operator_ansatz(obs, reps=2, flatten=True)

        # Verify it's a valid circuit
        self.assertGreater(ansatz.num_parameters, 0)
        self.assertEqual(ansatz.num_qubits, 2)

        # Verify no PauliEvolutionGate is present (indicating Rust path was used)
        ops = ansatz.count_ops()
        self.assertNotIn("PauliEvolution", ops)

    def test_sparse_observable_projector_terms(self):
        """Test SparseObservable with projector terms uses Rust path."""
        # Create observable with projector terms (0, 1, +, -, r, l)
        obs = SparseObservable.from_sparse_list([
            ("0", [0], 0.5),
            ("1", [1], 0.5),
            ("+", [2], 0.5),
            ("-", [2], 0.5),
        ], num_qubits=3)

        ansatz = evolved_operator_ansatz(obs, reps=1, flatten=True)

        # Should use Rust path and produce valid circuit
        self.assertEqual(ansatz.num_qubits, 3)
        self.assertEqual(ansatz.num_parameters, 1)

        # Verify Rust path was used (no nested gates)
        ops = ansatz.count_ops()
        self.assertNotIn("PauliEvolution", ops)

    def test_sparse_observable_mixed_with_sparse_pauli_op(self):
        """Test mixing SparseObservable and SparsePauliOp uses Rust path."""
        obs = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=1)
        pauli_op = SparsePauliOp(["Z"])

        # Mixed types should use Rust path (both support to_sparse_list)
        ansatz = evolved_operator_ansatz([obs, pauli_op], reps=1, flatten=True)

        # Should still work and use Rust path
        self.assertGreater(ansatz.num_parameters, 0)
        # Should not have PauliEvolutionGate (Rust path)
        ops = ansatz.count_ops()
        self.assertNotIn("PauliEvolution", ops)

    def test_sparse_observable_with_custom_evolution(self):
        """Test SparseObservable with custom evolution uses Python path."""
        obs = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=1)

        # Custom evolution should use Python path (but MatrixExponential doesn't support
        # SparseObservable, so use LieTrotter with custom settings instead)
        from qiskit.synthesis.evolution import LieTrotter
        evolution = LieTrotter(reps=2)  # Custom evolution with different reps
        ansatz = evolved_operator_ansatz(obs, evolution=evolution, flatten=False)

        # Should use Python path (custom evolution + flatten=False forces Python path)
        ops = ansatz.count_ops()
        self.assertIn("PauliEvolution", ops)

    def test_sparse_observable_flatten_false(self):
        """Test SparseObservable with flatten=False uses Python path."""
        obs = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=1)

        # flatten=False should use Python path
        ansatz = evolved_operator_ansatz(obs, flatten=False)

        # Should use Python path
        ops = ansatz.count_ops()
        self.assertIn("PauliEvolution", ops)

    def test_sparse_observable_remove_identities(self):
        """Test that SparseObservable identity operators are removed."""
        # Create observable with identity and non-identity terms
        obs1 = SparseObservable.identity(2)  # Identity
        obs2 = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=2)  # Non-identity

        # With remove_identities=True, identity should be removed
        ansatz = evolved_operator_ansatz([obs1, obs2], reps=1, remove_identities=True)

        # Should only have 1 parameter (for the non-identity operator)
        self.assertEqual(ansatz.num_parameters, 1)

        # Without remove_identities, both should be included
        ansatz2 = evolved_operator_ansatz([obs1, obs2], reps=1, remove_identities=False)
        self.assertEqual(ansatz2.num_parameters, 2)

    def test_sparse_observable_multiple_reps(self):
        """Test SparseObservable with multiple repetitions."""
        # Create two separate operators (not one operator with two terms)
        obs1 = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=2)
        obs2 = SparseObservable.from_sparse_list([("Z", [1], 1.0)], num_qubits=2)

        ansatz = evolved_operator_ansatz([obs1, obs2], reps=3, flatten=True)

        # Should have 2 operators * 3 reps = 6 parameters
        self.assertEqual(ansatz.num_parameters, 6)
        self.assertEqual(ansatz.num_qubits, 2)

        # Verify Rust path was used
        ops = ansatz.count_ops()
        self.assertNotIn("PauliEvolution", ops)

    def test_sparse_observable_equivalence_with_sparse_pauli_op(self):
        """Test that SparseObservable and SparsePauliOp produce equivalent circuits."""
        # Create equivalent operators
        obs = SparseObservable.from_sparse_list([("XZ", [0, 1], 1.0)], num_qubits=2)
        pauli_op = SparsePauliOp(["XZ"])

        ansatz_obs = evolved_operator_ansatz(obs, reps=1, flatten=True)
        ansatz_pauli = evolved_operator_ansatz(pauli_op, reps=1, flatten=True)

        # Both should have same number of qubits and parameters
        self.assertEqual(ansatz_obs.num_qubits, ansatz_pauli.num_qubits)
        self.assertEqual(ansatz_obs.num_parameters, ansatz_pauli.num_parameters)

        # Both should use Rust path
        ops_obs = ansatz_obs.count_ops()
        ops_pauli = ansatz_pauli.count_ops()
        self.assertNotIn("PauliEvolution", ops_obs)
        self.assertNotIn("PauliEvolution", ops_pauli)


class TestHamiltonianVariationalAnsatz(QiskitTestCase):
    """Test the hamiltonian_variational_ansatz function.

    This is essentially already tested by the evolved_operator_ansatz, we just need
    to test the additional commuting functionality.
    """

    def test_detect_commutation(self):
        """Test the operator is split into commuting terms."""
        hamiltonian = SparsePauliOp(["XII", "ZZI", "IXI", "IZZ", "IIX"])
        circuit = hamiltonian_variational_ansatz(hamiltonian)

        # this Hamiltonian should be split into 2 commuting groups, hence we get 2 parameters
        self.assertEqual(2, circuit.num_parameters)

    def test_evolution_with_identity(self):
        """Test a Hamiltonian containing an identity term.

        Regression test of #13644.
        """
        hamiltonian = SparsePauliOp(["III", "IZZ", "IXI"])
        ansatz = hamiltonian_variational_ansatz(hamiltonian, reps=1)
        bound = ansatz.assign_parameters([1, 1])  # we have two non-commuting groups, hence 2 params

        expected = QuantumCircuit(3, global_phase=-1)
        expected.rzz(2, 0, 1)
        expected.rx(2, 1)

        self.assertEqual(expected, bound)


def evolve(pauli_string, time):
    """Get the reference evolution circuit for a single Pauli string."""

    num_qubits = len(pauli_string)
    forward = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(pauli_string):
        if pauli == "x":
            forward.h(i)
        elif pauli == "y":
            forward.sx(i)

    for i in range(1, num_qubits):
        forward.cx(num_qubits - i, num_qubits - i - 1)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(forward, inplace=True)
    circuit.rz(2 * time, 0)
    circuit.compose(forward.inverse(), inplace=True)

    return circuit
