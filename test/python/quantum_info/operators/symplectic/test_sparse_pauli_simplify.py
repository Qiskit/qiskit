import numpy as np
import unittest
from qiskit.quantum_info import SparsePauliOp

class TestSparsePauliOpUnitary(unittest.TestCase):
    """
    Tests for the is_unitary method of SparsePauliOp, specifically
    addressing tolerance propagation to simplify().
    """

    def test_is_unitary_with_tolerance(self):
        """
        Test that is_unitary correctly respects input atol/rtol
        when simplifying the composed operator.
        """
        # This matrix 'a' is designed such that a @ a.conj().T is very close to identity,
        # but has very small off-diagonal elements and a small Z component in its SparsePauliOp
        # representation of the composed operator, which should be simplified to zero
        # when tolerance is applied correctly.
        a = np.array(
            [[-0.99801135 + 0.063036762j, 0.0000056710692 + 0.000081099635j],
             [0.0000056710610 + 0.000081099643j, -0.99707150 + 0.076473624j]]
        )

        # Convert the nearly unitary matrix to SparsePauliOp
        pauli_op = SparsePauliOp.from_operator(a)

        # Define tolerances that should be sufficient to simplify the small
        # non-identity components to zero.
        # These values are chosen based on the problem description's example.
        test_atol = 1e-5
        test_rtol = 1e-3

        # Before the fix (PR #14564), this would return False because
        # the compose().adjoint().simplify() call didn't use these tolerances,
        # leaving a small Z component that caused the check to fail.
        # After the fix, this should correctly return True.
        self.assertTrue(pauli_op.is_unitary(atol=test_atol, rtol=test_rtol),
                        f"Expected SparsePauliOp to be unitary with atol={test_atol}, rtol={test_rtol},"
                        " but is_unitary returned False.")

    def test_is_unitary_strict_tolerance(self):
        """
        Test that is_unitary still correctly identifies non-unitary operators
        with strict tolerances.
        """
        # A clearly non-unitary operator
        non_unitary_op = SparsePauliOp.from_list([('X', 1.0), ('Y', 0.1)])

        # With default or strict tolerances, this should be False
        self.assertFalse(non_unitary_op.is_unitary(),
                         "Expected non-unitary operator to be identified as such.")
        self.assertFalse(non_unitary_op.is_unitary(atol=1e-10, rtol=1e-10),
                         "Expected non-unitary operator to be identified as such with strict tolerances.")

# This block allows running the test directly if this file is executed.
if __name__ == '__main__':
    unittest.main()