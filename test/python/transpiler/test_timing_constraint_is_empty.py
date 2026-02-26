import unittest
from qiskit.transpiler.timing_constraints import TimingConstraints

#Validate the is_empty method of the TimingConstraints class
class TestTimingConstraintIsEmpty(unittest.TestCase):
    def test_is_empty(self):
        timing_constraints = TimingConstraints()
        print(timing_constraints.is_empty())
        self.assertTrue(timing_constraints.is_empty())

    def test_is_not_empty(self):
        timing_constraints = TimingConstraints(acquire_alignment=2)
        print(timing_constraints.is_empty())
        self.assertFalse(timing_constraints.is_empty())

if __name__ == "__main__":
    unittest.main()