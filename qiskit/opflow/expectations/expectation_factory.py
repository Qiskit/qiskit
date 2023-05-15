# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ExpectationFactory Class"""

import logging
from typing import Optional, Union

from qiskit import BasicAer
from qiskit.opflow.expectations.aer_pauli_expectation import AerPauliExpectation
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.expectations.matrix_expectation import MatrixExpectation
from qiskit.opflow.expectations.pauli_expectation import PauliExpectation
from qiskit.opflow.operator_base import OperatorBase
from qiskit.providers import Backend
from qiskit.utils.backend_utils import is_aer_qasm, is_statevector_backend
from qiskit.utils import QuantumInstance, optionals
from qiskit.utils.deprecation import deprecate_func

logger = logging.getLogger(__name__)


class ExpectationFactory:

    """Deprecated:  factory class for convenient automatic selection of an Expectation based on the
    Operator to be converted and backend used to sample the expectation value.
    """

    @staticmethod
    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def build(
        operator: OperatorBase,
        backend: Optional[Union[Backend, QuantumInstance]] = None,
        include_custom: bool = True,
    ) -> ExpectationBase:
        """
        A factory method for convenient automatic selection of an Expectation based on the
        Operator to be converted and backend used to sample the expectation value.

        Args:
            operator: The Operator whose expectation value will be taken.
            backend: The backend which will be used to sample the expectation value.
            include_custom: Whether the factory will include the (Aer) specific custom
                expectations if their behavior against the backend might not be as expected.
                For instance when using Aer qasm_simulator with paulis the Aer snapshot can
                be used but the outcome lacks shot noise and hence does not intuitively behave
                overall as people might expect when choosing a qasm_simulator. It is however
                fast as long as the more state vector like behavior is acceptable.

        Returns:
            The expectation algorithm which best fits the Operator and backend.

        Raises:
            ValueError: If operator is not of a composition for which we know the best Expectation
                method.
        """
        backend_to_check = backend.backend if isinstance(backend, QuantumInstance) else backend

        # pylint: disable=cyclic-import
        primitives = operator.primitive_strings()
        if primitives in ({"Pauli"}, {"SparsePauliOp"}):

            if backend_to_check is None:
                # If user has Aer but didn't specify a backend, use the Aer fast expectation
                if optionals.HAS_AER:
                    from qiskit_aer import AerSimulator

                    backend_to_check = AerSimulator()
                # If user doesn't have Aer, use statevector_simulator
                # for < 16 qubits, and qasm with warning for more.
                else:
                    if operator.num_qubits <= 16:
                        backend_to_check = BasicAer.get_backend("statevector_simulator")
                    else:
                        logger.warning(
                            "%d qubits is a very large expectation value. "
                            "Consider installing Aer to use "
                            "Aer's fast expectation, which will perform better here. We'll use "
                            "the BasicAer qasm backend for this expectation to avoid having to "
                            "construct the %dx%d operator matrix.",
                            operator.num_qubits,
                            2**operator.num_qubits,
                            2**operator.num_qubits,
                        )
                        backend_to_check = BasicAer.get_backend("qasm_simulator")

            # If the user specified Aer qasm backend and is using a
            # Pauli operator, use the Aer fast expectation if we are including such
            # custom behaviors.
            if is_aer_qasm(backend_to_check) and include_custom:
                return AerPauliExpectation()

            # If the user specified a statevector backend (either Aer or BasicAer),
            # use a converter to produce a
            # Matrix operator and compute using matmul
            elif is_statevector_backend(backend_to_check):
                if operator.num_qubits >= 16:
                    logger.warning(
                        "Note: Using a statevector_simulator with %d qubits can be very expensive. "
                        "Consider using the Aer qasm_simulator instead to take advantage of Aer's "
                        "built-in fast Pauli Expectation",
                        operator.num_qubits,
                    )
                return MatrixExpectation()

            # All other backends, including IBMQ, BasicAer QASM, go here.
            else:
                return PauliExpectation()

        elif primitives == {"Matrix"}:
            return MatrixExpectation()

        else:
            raise ValueError("Expectations of Mixed Operators not yet supported.")
