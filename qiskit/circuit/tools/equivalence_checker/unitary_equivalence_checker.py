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

from .base_equivalence_checker import BaseEquivalenceChecker, EquivalenceCheckerResult

class UnitaryEquivalenceChecker(BaseEquivalenceChecker):
    def __init__(self):
        super().__init__('unitary')
    
    def _run_checker(self, circ1, circ2, phase):
        from qiskit.quantum_info.operators import Operator
        equivalent = None
        success = True
        error_msg = ''

        ops = []
        circs = [circ1, circ2]

        for circ in circs:
            try:
                op = Operator(circ)
                ops.append(op)
            except Exception as e:
                error_msg += 'Circuit ' + circ.name + ' is invalid: ' + str(e) + '\n'
                success = False

        if success:
            try:
                if phase == 'equal':
                    equivalent = (ops[0] == ops[1])
                elif phase == 'up_to_global':
                    equivalent = (ops[0].equiv(ops[1]))
                else:
                    raise('Unrecognized phase criterion: ' + str(phase))
            except:
                error_msg = e
                success = False

        return EquivalenceCheckerResult(success, equivalent, error_msg)
