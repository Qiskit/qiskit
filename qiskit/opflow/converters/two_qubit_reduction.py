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

"""Z2 Symmetry Tapering Converter Class"""

import logging
from typing import List, Tuple, Union, cast

from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import Z2Symmetries
from qiskit.quantum_info import Pauli
from qiskit.utils.deprecation import deprecate_func

logger = logging.getLogger(__name__)


class TwoQubitReduction(ConverterBase):
    """
    Deprecated: Two qubit reduction converter which eliminates the central and last
    qubit in a list of Pauli that has diagonal operators (Z,I) at those positions.

    Chemistry specific method:
    It can be used to taper two qubits in parity and binary-tree mapped
    fermionic Hamiltonians when the spin orbitals are ordered in two spin
    sectors, (block spin order) according to the number of particles in the system.
    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, num_particles: Union[int, List[int], Tuple[int, int]]):
        """
        Args:
            num_particles: number of particles, if it is a list,
                           the first number is alpha and the second number if beta.
        """
        super().__init__()
        if isinstance(num_particles, (tuple, list)):
            num_alpha = num_particles[0]
            num_beta = num_particles[1]
        else:
            num_alpha = num_particles // 2
            num_beta = num_particles // 2

        par_1 = 1 if (num_alpha + num_beta) % 2 == 0 else -1
        par_2 = 1 if num_alpha % 2 == 0 else -1
        self._tapering_values = [par_2, par_1]

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """
        Converts the Operator to tapered one by Z2 symmetries.

        Args:
            operator: the operator
        Returns:
            A new operator whose qubit number is reduced by 2.
        """
        if not isinstance(operator, PauliSumOp):
            return operator

        operator = cast(PauliSumOp, operator)

        if operator.is_zero():
            logger.info(
                "Operator is empty, can not do two qubit reduction. Return the empty operator back."
            )
            return PauliSumOp.from_list([("I" * (operator.num_qubits - 2), 0)])

        num_qubits = operator.num_qubits
        last_idx = num_qubits - 1
        mid_idx = num_qubits // 2 - 1
        sq_list = [mid_idx, last_idx]

        # build symmetries, sq_paulis:
        symmetries, sq_paulis = [], []
        for idx in sq_list:
            pauli_str = ["I"] * num_qubits

            pauli_str[idx] = "Z"
            z_sym = Pauli("".join(pauli_str)[::-1])
            symmetries.append(z_sym)

            pauli_str[idx] = "X"
            sq_pauli = Pauli("".join(pauli_str)[::-1])
            sq_paulis.append(sq_pauli)

        z2_symmetries = Z2Symmetries(symmetries, sq_paulis, sq_list, self._tapering_values)
        return z2_symmetries.taper(operator)
