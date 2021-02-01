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

from time import time

class EquivalenceCheckerResult():
    def __init__(self, success, equivalent, time_taken,
                 circname1, circname2):
        self._success = success
        self._time_taken = time_taken
        self._equivalent = equivalent
        self._circname1 = circname1
        self._circname2 = circname2

class EquivalenceChecker():
    def run(self, circ1, circ2):
        from qiskit.quantum_info.operators import Operator
        is_equivalent = None
        success = True

        start = time()
        try:
            equivalent = (Operator(circ1) == Operator(circ2))
        except:
            success = False

        time_taken = time() - start
        return EquivalenceCheckerResult(success, equivalent, time_taken,
                                        circ1.name, circ2.name)
