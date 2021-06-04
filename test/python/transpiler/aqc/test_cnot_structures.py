# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests the set of functions for generating CNOT structures.
"""
print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import sys, os, traceback

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import numpy as np
import unittest
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network


# TODO: implement a proper test!


class TestCnotStructures(unittest.TestCase):
    def test_structures(self):
        print("WARNING: TODO: not implemented yet")


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
