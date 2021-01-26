from time import time
from qiskit.quantum_info.operators import Operator

class EquivalenceCheckerResult():
    def __init__(self, success, is_equivalent, time_taken,
                 circname1, circname2):
        self._success = success
        self._time_taken = time_taken
        self._is_equivalent = is_equivalent
        self._circname1 = circname1
        self._circname2 = circname2

class EquivalenceChecker():
    def run(circ1, circ2):
        is_equivalent = None
        success = True

        start = time()
        try:
            is_equivalent = (Operator(circ1) == Operator(circ2))
        except:
            success = False

        time_taken = time() - start
        return EquivalenceCheckerResult(success, is_equivalent, time_taken,
                                        circ1.name, circ2.name)
