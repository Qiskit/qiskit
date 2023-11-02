# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TaperedPauliSumOp Class and Z2Symmetries"""

import itertools
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union, cast

import numpy as np

from qiskit.circuit import ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.utils import commutator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.utils.deprecation import deprecate_func

logger = logging.getLogger(__name__)


class TaperedPauliSumOp(PauliSumOp):
    """Deprecated: Class for PauliSumOp after tapering"""

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: SparsePauliOp,
        z2_symmetries: "Z2Symmetries",
        coeff: Union[complex, ParameterExpression] = 1.0,
    ) -> None:
        """
        Args:
            primitive: The SparsePauliOp which defines the behavior of the underlying function.
            z2_symmetries: Z2 symmetries which the Operator has.
            coeff: A coefficient multiplying the primitive.

        Raises:
            TypeError: invalid parameters.
        """
        super().__init__(primitive, coeff)
        if not isinstance(z2_symmetries, Z2Symmetries):
            raise TypeError(
                f"Argument parameter z2_symmetries must be Z2Symmetries, not {type(z2_symmetries)}"
            )
        self._z2_symmetries = z2_symmetries

    @property
    def z2_symmetries(self) -> "Z2Symmetries":
        """
        Z2 symmetries which the Operator has.

        Returns:
            The Z2 Symmetries.
        """
        return self._z2_symmetries

    @property
    def settings(self) -> Dict:
        """Return operator settings."""
        return {
            "primitive": self._primitive,
            "z2_symmetries": self._z2_symmetries,
            "coeff": self._coeff,
        }

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        pauli_sum = PauliSumOp(self.primitive, self.coeff)
        return pauli_sum.assign_parameters(param_dict)


class Z2Symmetries:
    """Deprecated: Z2 Symmetries"""

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        symmetries: List[Pauli],
        sq_paulis: List[Pauli],
        sq_list: List[int],
        tapering_values: Optional[List[int]] = None,
        tol: float = 1e-14,
    ):
        """
        Args:
            symmetries: the list of Pauli objects representing the Z_2 symmetries
            sq_paulis: the list of single - qubit Pauli objects to construct the
                                     Clifford operators
            sq_list: the list of support of the single-qubit Pauli objects used to build
                                 the Clifford operators
            tapering_values: values determines the sector.
            tol: Tolerance threshold for ignoring real and complex parts of a coefficient.

        Raises:
            OpflowError: Invalid paulis
        """
        if len(symmetries) != len(sq_paulis):
            raise OpflowError(
                "Number of Z2 symmetries has to be the same as number of single-qubit pauli x."
            )

        if len(sq_paulis) != len(sq_list):
            raise OpflowError(
                "Number of single-qubit pauli x has to be the same as length of single-qubit list."
            )

        if tapering_values is not None:
            if len(sq_list) != len(tapering_values):
                raise OpflowError(
                    "The length of single-qubit list has "
                    "to be the same as length of tapering values."
                )

        self._symmetries = symmetries
        self._sq_paulis = sq_paulis
        self._sq_list = sq_list
        self._tapering_values = tapering_values
        self._tol = tol

    @property
    def tol(self):
        """Tolerance threshold for ignoring real and complex parts of a coefficient."""
        return self._tol

    @tol.setter
    def tol(self, value):
        """Set the tolerance threshold for ignoring real and complex parts of a coefficient."""
        self._tol = value

    @property
    def symmetries(self):
        """return symmetries"""
        return self._symmetries

    @property
    def sq_paulis(self):
        """returns sq paulis"""
        return self._sq_paulis

    @property
    def cliffords(self) -> List[PauliSumOp]:
        """
        Get clifford operators, build based on symmetries and single-qubit X.
        Returns:
            a list of unitaries used to diagonalize the Hamiltonian.
        """
        cliffords = [
            (PauliOp(pauli_symm) + PauliOp(sq_pauli)) / np.sqrt(2)
            for pauli_symm, sq_pauli in zip(self._symmetries, self._sq_paulis)
        ]
        return cliffords

    @property
    def sq_list(self):
        """returns sq list"""
        return self._sq_list

    @property
    def tapering_values(self):
        """returns tapering values"""
        return self._tapering_values

    @tapering_values.setter
    def tapering_values(self, new_value):
        """set tapering values"""
        self._tapering_values = new_value

    @property
    def settings(self) -> Dict:
        """Return operator settings."""
        return {
            "symmetries": self._symmetries,
            "sq_paulis": self._sq_paulis,
            "sq_list": self._sq_list,
            "tapering_values": self._tapering_values,
        }

    def __str__(self):
        ret = ["Z2 symmetries:"]
        ret.append("Symmetries:")
        for symmetry in self._symmetries:
            ret.append(symmetry.to_label())
        ret.append("Single-Qubit Pauli X:")
        for x in self._sq_paulis:
            ret.append(x.to_label())
        ret.append("Cliffords:")
        for c in self.cliffords:
            ret.append(str(c))
        ret.append("Qubit index:")
        ret.append(str(self._sq_list))
        ret.append("Tapering values:")
        if self._tapering_values is None:
            possible_values = [
                str(list(coeff)) for coeff in itertools.product([1, -1], repeat=len(self._sq_list))
            ]
            possible_values = ", ".join(x for x in possible_values)
            ret.append("  - Possible values: " + possible_values)
        else:
            ret.append(str(self._tapering_values))

        ret = "\n".join(ret)
        return ret

    def copy(self) -> "Z2Symmetries":
        """
        Get a copy of self.
        Returns:
            copy
        """
        return deepcopy(self)

    def is_empty(self) -> bool:
        """
        Check the z2_symmetries is empty or not.
        Returns:
            Empty or not
        """
        return self._symmetries == [] or self._sq_paulis == [] or self._sq_list == []

    # pylint: disable=invalid-name
    @classmethod
    def find_Z2_symmetries(cls, operator: PauliSumOp) -> "Z2Symmetries":
        """
        Finds Z2 Pauli-type symmetries of an Operator.

        Returns:
            a z2_symmetries object contains symmetries, single-qubit X, single-qubit list.
        """
        pauli_symmetries = []
        sq_paulis = []
        sq_list = []

        stacked_paulis = []

        if operator.is_zero():
            logger.info("Operator is empty.")
            return cls([], [], [], None)

        for pauli in operator:
            stacked_paulis.append(
                np.concatenate(
                    (pauli.primitive.paulis.x[0], pauli.primitive.paulis.z[0]), axis=0
                ).astype(int)
            )

        stacked_matrix = np.array(np.stack(stacked_paulis))
        symmetries = _kernel_F2(stacked_matrix)

        if not symmetries:
            logger.info("No symmetry is found.")
            return cls([], [], [], None)

        stacked_symmetries = np.stack(symmetries)
        symm_shape = stacked_symmetries.shape

        for row in range(symm_shape[0]):

            pauli_symmetries.append(
                Pauli(
                    (
                        stacked_symmetries[row, : symm_shape[1] // 2],
                        stacked_symmetries[row, symm_shape[1] // 2 :],
                    )
                )
            )

            stacked_symm_del = np.delete(stacked_symmetries, row, axis=0)
            for col in range(symm_shape[1] // 2):
                # case symmetries other than one at (row) have Z or I on col qubit
                Z_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (
                        stacked_symm_del[symm_idx, col] == 0
                        and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] in (0, 1)
                    ):
                        Z_or_I = False
                if Z_or_I:
                    if (
                        stacked_symmetries[row, col] == 1
                        and stacked_symmetries[row, col + symm_shape[1] // 2] == 0
                    ) or (
                        stacked_symmetries[row, col] == 1
                        and stacked_symmetries[row, col + symm_shape[1] // 2] == 1
                    ):
                        sq_paulis.append(
                            Pauli((np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        )
                        sq_paulis[row].z[col] = False
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row) have X or I on col qubit
                X_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (
                        stacked_symm_del[symm_idx, col] in (0, 1)
                        and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0
                    ):
                        X_or_I = False
                if X_or_I:
                    if (
                        stacked_symmetries[row, col] == 0
                        and stacked_symmetries[row, col + symm_shape[1] // 2] == 1
                    ) or (
                        stacked_symmetries[row, col] == 1
                        and stacked_symmetries[row, col + symm_shape[1] // 2] == 1
                    ):
                        sq_paulis.append(
                            Pauli((np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        )
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = False
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row)  have Y or I on col qubit
                Y_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (
                        (
                            stacked_symm_del[symm_idx, col] == 1
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 1
                        )
                        or (
                            stacked_symm_del[symm_idx, col] == 0
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0
                        )
                    ):
                        Y_or_I = False
                if Y_or_I:
                    if (
                        stacked_symmetries[row, col] == 0
                        and stacked_symmetries[row, col + symm_shape[1] // 2] == 1
                    ) or (
                        stacked_symmetries[row, col] == 1
                        and stacked_symmetries[row, col + symm_shape[1] // 2] == 0
                    ):
                        sq_paulis.append(
                            Pauli((np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        )
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

        return cls(pauli_symmetries, sq_paulis, sq_list, None)

    def convert_clifford(self, operator: PauliSumOp) -> OperatorBase:
        """This method operates the first part of the tapering.
        It converts the operator by composing it with the clifford unitaries defined in the current
        symmetry.

        Args:
            operator: to-be-tapered operator

        Returns:
            :class:`PauliSumOp` corresponding to the converted operator.

        Raises:
            OpflowError: Z2 symmetries, single qubit pauli and single qubit list cannot be empty

        """

        if not self._symmetries or not self._sq_paulis or not self._sq_list:
            raise OpflowError(
                "Z2 symmetries, single qubit pauli and single qubit list cannot be empty."
            )

        if not operator.is_zero():
            for clifford in self.cliffords:
                operator = cast(PauliSumOp, clifford @ operator @ clifford)
                operator = operator.reduce(atol=0)

        return operator

    def taper_clifford(self, operator: PauliSumOp) -> OperatorBase:
        """This method operates the second part of the tapering.
        This function assumes that the input operators have already been transformed using
        :meth:`convert_clifford`. The redundant qubits due to the symmetries are dropped and
        replaced by their two possible eigenvalues.
        The `tapering_values` will be stored into the resulted operator for a record.

        Args:
            operator: Partially tapered operator resulting from a call to :meth:`convert_clifford`

        Returns:
            If tapering_values is None: [:class:`PauliSumOp`]; otherwise, :class:`PauliSumOp`

        Raises:
            OpflowError: Z2 symmetries, single qubit pauli and single qubit list cannot be empty

        """

        if not self._symmetries or not self._sq_paulis or not self._sq_list:
            raise OpflowError(
                "Z2 symmetries, single qubit pauli and single qubit list cannot be empty."
            )
        # If the operator is zero then we can skip the following. We still need to taper the
        # operator to reduce its size i.e. the number of qubits so for example 0*"IIII" could
        # taper to 0*"II" when symmetries remove two qubits.
        if self._tapering_values is None:
            tapered_ops_list = [
                self._taper(operator, list(coeff))
                for coeff in itertools.product([1, -1], repeat=len(self._sq_list))
            ]
            tapered_ops: OperatorBase = ListOp(tapered_ops_list)
        else:
            tapered_ops = self._taper(operator, self._tapering_values)

        return tapered_ops

    def taper(self, operator: PauliSumOp) -> OperatorBase:
        """
        Taper an operator based on the z2_symmetries info and sector defined by `tapering_values`.
        The `tapering_values` will be stored into the resulted operator for a record.

        The tapering is a two-step algorithm which first converts the operator into a
        :class:`PauliSumOp` with same eigenvalues but where some qubits are only acted upon
        with the Pauli operators I or X.
        The number M of these redundant qubits is equal to the number M of identified symmetries.

        The second step of the reduction consists in replacing these qubits with the possible
        eigenvalues of the corresponding Pauli X, giving 2^M new operators with M less qubits.
        If an eigenvalue sector was previously identified for the solution, then this reduces to
        1 new operator with M less qubits.

        Args:
            operator: the to-be-tapered operator

        Returns:
            If tapering_values is None: [:class:`PauliSumOp`]; otherwise, :class:`PauliSumOp`

        Raises:
            OpflowError: Z2 symmetries, single qubit pauli and single qubit list cannot be empty

        """

        if not self._symmetries or not self._sq_paulis or not self._sq_list:
            raise OpflowError(
                "Z2 symmetries, single qubit pauli and single qubit list cannot be empty."
            )

        converted_ops = self.convert_clifford(operator)
        tapered_ops = self.taper_clifford(converted_ops)

        return tapered_ops

    def _taper(self, op: PauliSumOp, curr_tapering_values: List[int]) -> OperatorBase:
        pauli_list = []
        for pauli_term in op:
            coeff_out = pauli_term.primitive.coeffs[0]
            for idx, qubit_idx in enumerate(self._sq_list):
                if (
                    pauli_term.primitive.paulis.z[0, qubit_idx]
                    or pauli_term.primitive.paulis.x[0, qubit_idx]
                ):
                    coeff_out = curr_tapering_values[idx] * coeff_out
            z_temp = np.delete(pauli_term.primitive.paulis.z[0].copy(), np.asarray(self._sq_list))
            x_temp = np.delete(pauli_term.primitive.paulis.x[0].copy(), np.asarray(self._sq_list))
            pauli_list.append((Pauli((z_temp, x_temp)).to_label(), coeff_out))

        spo = SparsePauliOp.from_list(pauli_list).simplify(atol=0.0)
        spo = spo.chop(self.tol)
        z2_symmetries = self.copy()
        z2_symmetries.tapering_values = curr_tapering_values

        return TaperedPauliSumOp(spo, z2_symmetries)

    def consistent_tapering(self, operator: PauliSumOp) -> OperatorBase:
        """
        Tapering the `operator` with the same manner of how this tapered operator
        is created. i.e., using the same Cliffords and tapering values.

        Args:
            operator: the to-be-tapered operator

        Returns:
            The tapered operator

        Raises:
            OpflowError: The given operator does not commute with the symmetry
        """
        for symmetry in self._symmetries:
            commutator_op = cast(PauliSumOp, commutator(operator, PauliOp(symmetry)))
            if not commutator_op.is_zero():
                raise OpflowError(
                    "The given operator does not commute with the symmetry, can not taper it."
                )

        return self.taper(operator)

    def __eq__(self, other: object) -> bool:
        """
        Overload `==` operation to evaluate equality between Z2Symmetries.

        Args:
            other: The `Z2Symmetries` to compare to self.

        Returns:
            A bool equal to the equality of self and other.
        """
        if not isinstance(other, Z2Symmetries):
            return False

        return (
            self.symmetries == other.symmetries
            and self.sq_paulis == other.sq_paulis
            and self.sq_list == other.sq_list
            and self.tapering_values == other.tapering_values
        )


def _kernel_F2(matrix_in) -> List[np.ndarray]:  # pylint: disable=invalid-name
    """
    Computes the kernel of a binary matrix on the binary finite field
    Args:
        matrix_in (numpy.ndarray): binary matrix
    Returns:
        The list of kernel vectors
    """
    size = matrix_in.shape
    kernel = []
    matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
    matrix_in_id_ech = (_row_echelon_F2(matrix_in_id.transpose())).transpose()

    for col in range(size[1]):
        if np.array_equal(
            matrix_in_id_ech[0 : size[0], col], np.zeros(size[0])
        ) and not np.array_equal(matrix_in_id_ech[size[0] :, col], np.zeros(size[1])):
            kernel.append(matrix_in_id_ech[size[0] :, col])

    return kernel


def _row_echelon_F2(matrix_in) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Computes the row Echelon form of a binary matrix on the binary finite field
    Args:
        matrix_in (numpy.ndarray): binary matrix
    Returns:
        Matrix_in in Echelon row form
    """
    size = matrix_in.shape

    for i in range(size[0]):
        pivot_index = 0
        for j in range(size[1]):
            if matrix_in[i, j] == 1:
                pivot_index = j
                break
        for k in range(size[0]):
            if k != i and matrix_in[k, pivot_index] == 1:
                matrix_in[k, :] = np.mod(matrix_in[k, :] + matrix_in[i, :], 2)

    matrix_out_temp = deepcopy(matrix_in)
    indices = []
    matrix_out = np.zeros(size)

    for i in range(size[0] - 1):
        if np.array_equal(matrix_out_temp[i, :], np.zeros(size[1])):
            indices.append(i)
    for row in np.sort(indices)[::-1]:
        matrix_out_temp = np.delete(matrix_out_temp, (row), axis=0)

    matrix_out[0 : size[0] - len(indices), :] = matrix_out_temp
    matrix_out = matrix_out.astype(int)

    return matrix_out
