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

class EquivalenceCheckerResult():
    def __init__(self, success, equivalent, time_taken,
                 circname1, circname2, error_msg):
        self.success = success
        self.time_taken = time_taken
        self.equivalent = equivalent
        self.circname1 = circname1
        self.circname2 = circname2
        self.error_msg = error_msg

class BaseEquivalenceChecker(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def run(self, circ1, circ2):
        pass
