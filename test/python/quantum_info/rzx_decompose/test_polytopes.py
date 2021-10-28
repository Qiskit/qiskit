"""
test/python/quantum_info/rzx_decompose/test_polytopes.py

Tests for qiskit-terra/qiskit/quantum_info/synthesis/rzx_decompose/xx_polytope.py .
"""

import random
import unittest

import ddt
import numpy as np
from numpy import pi

from qiskit.quantum_info.synthesis.rzx_decompose.polytopes import XXPolytope

EPSILON = 0.001


@ddt.ddt
class TestMonodromyXXPolytope(unittest.TestCase):
    """Check specialized XX polytope routines."""

    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(seed)
        np.random.seed(seed)

    @ddt.data(
        ([pi / 4, pi / 4, pi / 4], [0.52359878, 0.39269908, 0.31415927]),
        ([pi / 6, pi / 8, pi / 10], [pi / 6, pi / 8, pi / 10]),
        ([pi / 4, pi / 4, 0], [0.615228561, 0.615228561, 0]),
        ([pi / 4, pi / 8, 0], [pi / 4, pi / 8, 0]),
    )
    @ddt.unpack
    def test_nearest(self, offbody, expected):
        """Check that the nearest point calculator recovers some known cases."""
        polytope = XXPolytope.from_strengths(pi / 6, pi / 8, pi / 10)
        result = polytope.nearest(np.array(offbody))
        print(offbody, result)
        self.assertTrue(np.all(np.abs(np.array(expected) - result) < EPSILON))

    def test_add_strengths(self):
        """
        Tests that adding new strengths to an existing XXPolytope is equivalent to forming the
        appropriate XXPolytope from scratch.
        """
        strengths = [random.random() for _ in range(4)]
        small_polytope = XXPolytope.from_strengths(*strengths[:-1])
        large_polytope = XXPolytope.from_strengths(*strengths)
        self.assertEqual(small_polytope.add_strength(strengths[-1]), large_polytope)
