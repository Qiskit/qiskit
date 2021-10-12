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
A wrapper function to different equivalence checkers
"""

from .unitary_equivalence_checker import UnitaryEquivalenceChecker


def equivalence_checker(circ1, circ2, method, **options):
    """
    A wrapper function to different equivalence checkers

    Args:
        circ1 (QuantumCircuit): First circuit to check.
        circ2 (QuantumCircuit): Second circuit to check.
        method (str): checker method
        options: parameters of the specific checker

    Returns:
        EquivalenceCheckerResult: result of the equivalence check.

    Raises:
        ValueError: if `method` is not one of a set of recognized methods
    """

    if method == "unitary":
        checker = UnitaryEquivalenceChecker(**options)
        return checker.run(circ1, circ2)
    else:
        raise ValueError("Unknown checker method: " + method)
