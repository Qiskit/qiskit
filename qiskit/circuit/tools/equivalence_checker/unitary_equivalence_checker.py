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

from qiskit.exceptions import QiskitError
from .base_equivalence_checker import BaseEquivalenceChecker, EquivalenceCheckerResult

class UnitaryEquivalenceChecker(BaseEquivalenceChecker):
    def __init__(self, simulator, name='unitary', external_backend=None, **backend_options):
        super().__init__(name)
        self.simulator = simulator
        self.backend_options = backend_options

        if simulator == 'external':
            self.backend = external_backend
        elif simulator == 'aer':
            try:
                from qiskit.providers.aer import UnitarySimulator
                self.backend = UnitarySimulator()
                self.backend.set_options(**backend_options)
            except ImportError:
                raise QiskitError('Could not import the Aer simulator')
        elif simulator == 'quantum_info':
            self.backend = None
        else:
            raise QiskitError('Unrecognized simulator option: ' + str(self.simulator))
    
    def _run_checker(self, circ1, circ2, phase):
        # importing here to avoid circular imports
        from qiskit.quantum_info.operators import Operator
        from qiskit.quantum_info.operators.predicates import is_identity_matrix
        from qiskit.compiler import transpile, assemble
        
        equivalent = None
        success = True
        error_msg = None

        try:
            circ = circ1 + circ2.inverse()
            circ = transpile(circ, self.backend)

            if self.simulator == 'quantum_info':
                op = Operator(circ)
            else:
                backend_res = self.backend.run(assemble(circ), shots=1, **self.backend_options).result()
                if backend_res.results[0].success:
                    op = backend_res.get_unitary(circ)
                else:
                    raise QiskitError(backend_res.results[0].status)

            if phase == 'equal':
                ignore_phase = False
            elif phase == 'up_to_global':
                ignore_phase = True
            else:
                raise QiskitError('Unrecognized phase criterion: ' + str(phase))

            # TODO: This can be made more efficient, because when checking whether
            # a unitary matrix is the identity, it suffices to check only the diagonal
            equivalent = is_identity_matrix(op, ignore_phase)
                
        except QiskitError as e:
            error_msg = str(e)
            success = False

        return EquivalenceCheckerResult(success, equivalent, error_msg)
