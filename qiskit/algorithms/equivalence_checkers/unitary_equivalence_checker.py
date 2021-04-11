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

"""UnitaryEquivalenceChecker class"""

from qiskit.exceptions import QiskitError
from .base_equivalence_checker import BaseEquivalenceChecker, EquivalenceCheckerResult


class UnitaryEquivalenceChecker(BaseEquivalenceChecker):
    """
    A tool to check equivalence of two circuits.
    The tool computes the unitary operator of a concatenation of one circuit with the
    inverse of the other circuit, and compares it to the identity.
    The computation of the unitary can be done either by the quantum_info module,
    by the Aer simulator, or by an external (non-Qiskit) simulator.
    The comparison can either ignore or not ignore global phase.
    """

    def __init__(self, simulator, name='unitary', external_backend=None, **backend_options):
        """
        Args:
            simulator (str): The type of simulator to compute the unitary.
                Options are: 'quantum_info', 'aer', or 'external'.
            name (str): The checker's name.
            external_backend (BaseBackend): The backend to run,  when `simulator` is 'external'.
            backend_options: Options to pass to the backend, when `simulator` is 'aer'
                or 'external'.

        Raises:
            QiskitError:
                If `simulator` is 'aer' but Aer is not installed.
                If `simulator` ia different from 'quantum_info', 'aer', 'external'.
        """

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

    # pylint: disable=arguments-differ
    def _run_checker(self, circ1, circ2, phase):
        """
        Check if circuits are equivalent.

        Args:
            circ1 (QuantumCircuit): First circuit to check.
            circ2 (QuantumCircuit): Second circuit to check.
            phase (str): Options are 'global' - ignoring global phase;
                or 'equal' - not ignoring global phase.

        Returns:
            EquivalenceCheckerResult: result of the equivalence check.

        Raises:
            QiskitError:
                If unitary creation fails (e.g., one of the circuit contains measurements,
                    or circuits are too large).
                If `phase` is not one of 'equal', 'global'.
        """

        # importing here to avoid circular imports
        from qiskit.quantum_info.operators import Operator
        from qiskit.quantum_info.operators.predicates import is_identity_matrix
        from qiskit.compiler import transpile, assemble

        equivalent = None
        success = True
        error_msg = None

        try:
            circ = circ1.compose(circ2.inverse())
            # Optimize the circuit before creating the unitary
            circ = transpile(circ, self.backend)

            if self.simulator == 'quantum_info':
                op = Operator(circ)
            else:
                backend_res = self.backend.run(assemble(circ), shots=1,
                                               **self.backend_options).result()
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

        # The broad class of Exception is required,
        # for example when the circuit is large,
        # then the error comes from a non-Qiskit package.
        # pylint: disable=broad-except
        except Exception as exc:
            error_msg = str(exc)
            success = False

        return EquivalenceCheckerResult(success, equivalent, error_msg)
