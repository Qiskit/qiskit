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

from abc import ABC, abstractmethod
from inspect import signature
from time import time


class EquivalenceCheckerResult:
    def __init__(self, success, equivalent,
                 circname1, circname2, error_msg):
        self.success = success
        self.equivalent = equivalent
        self.circname1 = circname1
        self.circname2 = circname2
        self.error_msg = error_msg

class BaseEquivalenceChecker(ABC):
    def __init__(self, name):
        self.name = name
    
    def run(self, circ1, circ2, **kwargs):
        start = time()        
        res = self._run_checker(circ1, circ2, **kwargs)
        time_taken = time() - start
        res.time_taken = time_taken
        
        return res

    @abstractmethod
    def _run_checker(self, circ1, circ2, **kwargs):
        pass
