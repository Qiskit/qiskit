# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PauliBasisChange Class """

from functools import partial, reduce
from typing import Callable, List, Optional, Tuple, Union, cast

import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.operator_globals import H, I, S
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.quantum_info import Pauli


class PauliBasisChange(ConverterBase):
    r"""
    Converter for changing Paulis into other bases. By default, the diagonal basis
    composed only of Pauli {Z, I}^n is used as the destination basis to which to convert.
    Meaning, if a Pauli containing X or Y terms is passed in, which cannot be
    sampled or evolved natively on some Quantum hardware, the Pauli can be replaced by a
    composition of a change of basis circuit and a Pauli composed of only Z
    and I terms (diagonal), which can be evolved or sampled natively on the Quantum
    hardware.

    The replacement function determines how the ``PauliOps`` should be replaced by their computed
    change-of-basis ``CircuitOps`` and destination ``PauliOps``. Several convenient out-of-the-box
    replacement functions have been added as static methods, such as ``measurement_replacement_fn``.

    This class uses the typical basis change method found in most Quantum Computing textbooks
    (such as on page 210 of Nielsen and Chuang's, "Quantum Computation and Quantum Information",
    ISBN: 978-1-107-00217-3), which involves diagonalizing the single-qubit Paulis with H and S†
    gates, mapping the eigenvectors of the diagonalized origin Pauli to the diagonalized
    destination Pauli using CNOTS, and then de-diagonalizing any single qubit Paulis to their
    non-diagonal destination values. Many other methods are possible, as well as variations on
    this method, such as the placement of the CNOT chains.
    """

    def __init__(
        self,
        destination_basis: Optional[Union[Pauli, PauliOp]] = None,
        traverse: bool = True,
        replacement_fn: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            destination_basis: The Pauli into the basis of which the operators
                will be converted. If None is specified, the destination basis will be the
                diagonal ({I, Z}^n) basis requiring only single qubit rotations.
            traverse: If true and the operator passed into convert contains sub-Operators,
                such as ListOp, traverse the Operator and apply the conversion to every
                applicable sub-operator within it.
            replacement_fn: A function specifying what to do with the basis-change
                ``CircuitOp`` and destination ``PauliOp`` when converting an Operator and
                replacing converted values. By default, this will be

                    1) For StateFns (or Measurements): replacing the StateFn with
                       ComposedOp(StateFn(d), c) where c is the conversion circuit and d is the
                       destination Pauli, so the overall beginning and ending operators are
                       equivalent.

                    2) For non-StateFn Operators: replacing the origin p with c·d·c†, where c
                       is the conversion circuit and d is the destination, so the overall
                       beginning and ending operators are equivalent.

        """
        if destination_basis is not None:
            self.destination = destination_basis  # type: ignore
        else:
            self._destination = None  # type: Optional[PauliOp]
        self._traverse = traverse
        self._replacement_fn = replacement_fn or PauliBasisChange.operator_replacement_fn

    @property
    def destination(self) -> Optional[PauliOp]:
        r"""
        The destination ``PauliOp``, or ``None`` if using the default destination, the diagonal
        basis.
        """
        return self._destination

    @destination.setter
    def destination(self, dest: Union[Pauli, PauliOp]) -> None:
        r"""
        The destination ``PauliOp``, or ``None`` if using the default destination, the diagonal
        basis.
        """
        if isinstance(dest, Pauli):
            dest = PauliOp(dest)

        if not isinstance(dest, PauliOp):
            raise TypeError(
                "PauliBasisChange can only convert into Pauli bases, " "not {}.".format(type(dest))
            )
        self._destination = dest

    # TODO see whether we should make this performant by handling ListOps of Paulis later.
    # pylint: disable=too-many-return-statements
    def convert(self, operator: OperatorBase) -> OperatorBase:
        r"""
        Given a ``PauliOp``, or an Operator containing ``PauliOps`` if ``_traverse`` is True,
        converts each Pauli into the basis specified by self._destination and a
        basis-change-circuit, calls ``replacement_fn`` with these two Operators, and replaces
        the ``PauliOps`` with the output of ``replacement_fn``. For example, for the built-in
        ``operator_replacement_fn`` below, each PauliOp p will be replaced by the composition
        of the basis-change Clifford ``CircuitOp`` c with the destination PauliOp d and c†,
        such that p = c·d·c†, up to global phase.

        Args:
            operator: The Operator to convert.

        Returns:
            The converted Operator.

        """
        if (
            isinstance(operator, OperatorStateFn)
            and isinstance(operator.primitive, PauliSumOp)
            and operator.primitive.grouping_type == "TPB"
        ):
            primitive = operator.primitive.primitive.copy()
            origin_x = reduce(np.logical_or, primitive.table.X)
            origin_z = reduce(np.logical_or, primitive.table.Z)
            origin_pauli = Pauli((origin_z, origin_x))
            cob_instr_op, _ = self.get_cob_circuit(origin_pauli)
            primitive.table.Z = np.logical_or(primitive.table.X, primitive.table.Z)
            primitive.table.X = False
            dest_pauli_sum_op = PauliSumOp(primitive, coeff=operator.coeff, grouping_type="TPB")
            return self._replacement_fn(cob_instr_op, dest_pauli_sum_op)

        if (
            isinstance(operator, OperatorStateFn)
            and isinstance(operator.primitive, SummedOp)
            and all(
                isinstance(op, PauliSumOp) and op.grouping_type == "TPB"
                for op in operator.primitive.oplist
            )
        ):
            sf_list: List[OperatorBase] = [
                StateFn(op, is_measurement=operator.is_measurement)
                for op in operator.primitive.oplist
            ]
            listop_of_statefns = SummedOp(oplist=sf_list, coeff=operator.coeff)
            return listop_of_statefns.traverse(self.convert)

        if isinstance(operator, OperatorStateFn) and isinstance(operator.primitive, PauliSumOp):
            operator = OperatorStateFn(
                operator.primitive.to_pauli_op(),
                coeff=operator.coeff,
                is_measurement=operator.is_measurement,
            )

        if isinstance(operator, PauliSumOp):
            operator = operator.to_pauli_op()

        if isinstance(operator, (Pauli, PauliOp)):
            cob_instr_op, dest_pauli_op = self.get_cob_circuit(operator)
            return self._replacement_fn(cob_instr_op, dest_pauli_op)
        if isinstance(operator, StateFn) and "Pauli" in operator.primitive_strings():
            # If the StateFn/Meas only contains a Pauli, use it directly.
            if isinstance(operator.primitive, PauliOp):
                cob_instr_op, dest_pauli_op = self.get_cob_circuit(operator.primitive)
                return self._replacement_fn(cob_instr_op, dest_pauli_op * operator.coeff)
            # TODO make a canonical "distribute" or graph swap as method in ListOp?
            elif operator.primitive.distributive:
                if operator.primitive.abelian:
                    origin_pauli = self.get_tpb_pauli(operator.primitive)
                    cob_instr_op, _ = self.get_cob_circuit(origin_pauli)
                    diag_ops: List[OperatorBase] = [
                        self.get_diagonal_pauli_op(op) for op in operator.primitive.oplist
                    ]
                    dest_pauli_op = operator.primitive.__class__(
                        diag_ops, coeff=operator.coeff, abelian=True
                    )
                    return self._replacement_fn(cob_instr_op, dest_pauli_op)
                else:
                    sf_list = [
                        StateFn(op, is_measurement=operator.is_measurement)
                        for op in operator.primitive.oplist
                    ]
                    listop_of_statefns = operator.primitive.__class__(
                        oplist=sf_list, coeff=operator.coeff
                    )
                    return listop_of_statefns.traverse(self.convert)

        elif (
            isinstance(operator, ListOp)
            and self._traverse
            and "Pauli" in operator.primitive_strings()
        ):
            # If ListOp is abelian we can find a single post-rotation circuit
            # for the whole set. For now,
            # assume operator can only be abelian if all elements are
            # Paulis (enforced in AbelianGrouper).
            if operator.abelian:
                origin_pauli = self.get_tpb_pauli(operator)
                cob_instr_op, _ = self.get_cob_circuit(origin_pauli)
                oplist = cast(List[PauliOp], operator.oplist)
                diag_ops = [self.get_diagonal_pauli_op(op) for op in oplist]
                dest_list_op = operator.__class__(diag_ops, coeff=operator.coeff, abelian=True)
                return self._replacement_fn(cob_instr_op, dest_list_op)
            else:
                return operator.traverse(self.convert)

        return operator

    @staticmethod
    def measurement_replacement_fn(
        cob_instr_op: PrimitiveOp, dest_pauli_op: Union[PauliOp, PauliSumOp, ListOp]
    ) -> OperatorBase:
        r"""
        A built-in convenience replacement function which produces measurements
        isomorphic to an ``OperatorStateFn`` measurement holding the origin ``PauliOp``.

        Args:
            cob_instr_op: The basis-change ``CircuitOp``.
            dest_pauli_op: The destination Pauli type operator.

        Returns:
            The ``~StateFn @ CircuitOp`` composition equivalent to a measurement by the original
            ``PauliOp``.
        """
        return ComposedOp([StateFn(dest_pauli_op, is_measurement=True), cob_instr_op])

    @staticmethod
    def statefn_replacement_fn(
        cob_instr_op: PrimitiveOp, dest_pauli_op: Union[PauliOp, PauliSumOp, ListOp]
    ) -> OperatorBase:
        r"""
        A built-in convenience replacement function which produces state functions
        isomorphic to an ``OperatorStateFn`` state function holding the origin ``PauliOp``.

        Args:
            cob_instr_op: The basis-change ``CircuitOp``.
            dest_pauli_op: The destination Pauli type operator.

        Returns:
            The ``~CircuitOp @ StateFn`` composition equivalent to a state function defined by the
            original ``PauliOp``.
        """
        return ComposedOp([cob_instr_op.adjoint(), StateFn(dest_pauli_op)])

    @staticmethod
    def operator_replacement_fn(
        cob_instr_op: PrimitiveOp, dest_pauli_op: Union[PauliOp, PauliSumOp, ListOp]
    ) -> OperatorBase:
        r"""
        A built-in convenience replacement function which produces Operators
        isomorphic to the origin ``PauliOp``.

        Args:
            cob_instr_op: The basis-change ``CircuitOp``.
            dest_pauli_op: The destination ``PauliOp``.

        Returns:
            The ``~CircuitOp @ PauliOp @ CircuitOp`` composition isomorphic to the
            original ``PauliOp``.
        """
        return ComposedOp([cob_instr_op.adjoint(), dest_pauli_op, cob_instr_op])

    def get_tpb_pauli(self, list_op: ListOp) -> Pauli:
        r"""
        Gets the Pauli (not ``PauliOp``!) whose diagonalizing single-qubit rotations is a
        superset of the diagonalizing single-qubit rotations for each of the Paulis in
        ``list_op``. TPB stands for `Tensor Product Basis`.

        Args:
             list_op: the :class:`ListOp` whose TPB Pauli to return.

        Returns:
             The TBP Pauli.

        """
        oplist = cast(List[PauliOp], list_op.oplist)
        origin_z = reduce(np.logical_or, [p_op.primitive.z for p_op in oplist])
        origin_x = reduce(np.logical_or, [p_op.primitive.x for p_op in oplist])
        return Pauli((origin_z, origin_x))

    def get_diagonal_pauli_op(self, pauli_op: PauliOp) -> PauliOp:
        """Get the diagonal ``PualiOp`` to which ``pauli_op`` could be rotated with only
        single-qubit operations.

        Args:
            pauli_op: The ``PauliOp`` whose diagonal to compute.

        Returns:
            The diagonal ``PauliOp``.
        """
        return PauliOp(
            Pauli(
                (
                    np.logical_or(pauli_op.primitive.z, pauli_op.primitive.x),
                    [False] * pauli_op.num_qubits,
                )
            ),
            coeff=pauli_op.coeff,
        )

    def get_diagonalizing_clifford(self, pauli: Union[Pauli, PauliOp]) -> OperatorBase:
        r"""
        Construct a ``CircuitOp`` with only single-qubit gates which takes the eigenvectors
        of ``pauli`` to eigenvectors composed only of \|0⟩ and \|1⟩ tensor products. Equivalently,
        finds the basis-change circuit to take ``pauli`` to a diagonal ``PauliOp`` composed only
        of Z and I tensor products.

        Note, underlying Pauli bits are in Qiskit endianness, so we need to reverse before we
        begin composing with Operator flow.

        Args:
            pauli: the ``Pauli`` or ``PauliOp`` to whose diagonalizing circuit to compute.

        Returns:
            The diagonalizing ``CircuitOp``.

        """
        if isinstance(pauli, PauliOp):
            pauli = pauli.primitive

        tensorall = cast(
            Callable[[List[PrimitiveOp]], PrimitiveOp], partial(reduce, lambda x, y: x.tensor(y))
        )

        y_to_x_origin = tensorall(
            [S if has_y else I for has_y in reversed(np.logical_and(pauli.x, pauli.z))]
        ).adjoint()
        x_to_z_origin = tensorall([H if has_x else I for has_x in reversed(pauli.x)])
        return x_to_z_origin.compose(y_to_x_origin)

    def pad_paulis_to_equal_length(
        self, pauli_op1: PauliOp, pauli_op2: PauliOp
    ) -> Tuple[PauliOp, PauliOp]:
        r"""
        If ``pauli_op1`` and ``pauli_op2`` do not act over the same number of qubits, pad
        identities to the end of the shorter of the two so they are of equal length. Padding is
        applied to the end of the Paulis. Note that the Terra represents Paulis in big-endian
        order, so this will appear as padding to the beginning of the Pauli x and z bit arrays.

        Args:
            pauli_op1: A pauli_op to possibly pad.
            pauli_op2: A pauli_op to possibly pad.

        Returns:
            A tuple containing the padded PauliOps.

        """
        num_qubits = max(pauli_op1.num_qubits, pauli_op2.num_qubits)
        pauli_1, pauli_2 = pauli_op1.primitive, pauli_op2.primitive

        # Padding to the end of the Pauli, but remember that Paulis are in reverse endianness.
        if not len(pauli_1.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_1.z)
            pauli_1 = Pauli(
                (
                    ([False] * missing_qubits) + pauli_1.z.tolist(),
                    ([False] * missing_qubits) + pauli_1.x.tolist(),
                )
            )
        if not len(pauli_2.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_2.z)
            pauli_2 = Pauli(
                (
                    ([False] * missing_qubits) + pauli_2.z.tolist(),
                    ([False] * missing_qubits) + pauli_2.x.tolist(),
                )
            )

        return PauliOp(pauli_1, coeff=pauli_op1.coeff), PauliOp(pauli_2, coeff=pauli_op2.coeff)

    def construct_cnot_chain(self, diag_pauli_op1: PauliOp, diag_pauli_op2: PauliOp) -> PrimitiveOp:
        r"""
        Construct a ``CircuitOp`` (or ``PauliOp`` if equal to the identity) which takes the
        eigenvectors of ``diag_pauli_op1`` to the eigenvectors of ``diag_pauli_op2``,
        assuming both are diagonal (or performing this operation on their diagonalized Paulis
        implicitly if not). This works by the insight that the eigenvalue of a diagonal Pauli's
        eigenvector is equal to or -1 if the parity is 1 and 1 if the parity is 0, or
        1 - (2 * parity). Therefore, using CNOTs, we can write the parity of diag_pauli_op1's
        significant bits onto some qubit, and then write out that parity onto diag_pauli_op2's
        significant bits.

        Args:
            diag_pauli_op1: The origin ``PauliOp``.
            diag_pauli_op2: The destination ``PauliOp``.

        Return:
            The ``PrimitiveOp`` performs the mapping.
        """
        # TODO be smarter about connectivity and actual distance between pauli and destination
        # TODO be smarter in general

        pauli_1 = (
            diag_pauli_op1.primitive if isinstance(diag_pauli_op1, PauliOp) else diag_pauli_op1
        )
        pauli_2 = (
            diag_pauli_op2.primitive if isinstance(diag_pauli_op2, PauliOp) else diag_pauli_op2
        )
        origin_sig_bits = np.logical_or(pauli_1.z, pauli_1.x)
        destination_sig_bits = np.logical_or(pauli_2.z, pauli_2.x)
        num_qubits = max(len(pauli_1.z), len(pauli_2.z))

        sig_equal_sig_bits = np.logical_and(origin_sig_bits, destination_sig_bits)
        non_equal_sig_bits = np.logical_not(origin_sig_bits == destination_sig_bits)
        # Equivalent to np.logical_xor(origin_sig_bits, destination_sig_bits)

        if not any(non_equal_sig_bits):
            return I ^ num_qubits

        # I am deeply sorry for this code, but I don't know another way to do it.
        sig_in_origin_only_indices = np.extract(
            np.logical_and(non_equal_sig_bits, origin_sig_bits), np.arange(num_qubits)
        )
        sig_in_dest_only_indices = np.extract(
            np.logical_and(non_equal_sig_bits, destination_sig_bits), np.arange(num_qubits)
        )

        if len(sig_in_origin_only_indices) > 0 and len(sig_in_dest_only_indices) > 0:
            origin_anchor_bit = min(sig_in_origin_only_indices)
            dest_anchor_bit = min(sig_in_dest_only_indices)
        else:
            # Set to lowest equal bit
            origin_anchor_bit = min(np.extract(sig_equal_sig_bits, np.arange(num_qubits)))
            dest_anchor_bit = origin_anchor_bit

        cnots = QuantumCircuit(num_qubits)
        # Step 3) Take the indices of bits which are sig_bits in
        # pauli but but not in dest, and cnot them to the pauli anchor.
        for i in sig_in_origin_only_indices:
            if not i == origin_anchor_bit:
                cnots.cx(i, origin_anchor_bit)

        # Step 4)
        if not origin_anchor_bit == dest_anchor_bit:
            cnots.swap(origin_anchor_bit, dest_anchor_bit)

        # Need to do this or a Terra bug sometimes flips cnots. No time to investigate.
        cnots.i(0)

        # Step 6)
        for i in sig_in_dest_only_indices:
            if not i == dest_anchor_bit:
                cnots.cx(i, dest_anchor_bit)

        return PrimitiveOp(cnots)

    def get_cob_circuit(self, origin: Union[Pauli, PauliOp]) -> Tuple[PrimitiveOp, PauliOp]:
        r"""
        Construct an Operator which maps the +1 and -1 eigenvectors
        of the origin Pauli to the +1 and -1 eigenvectors of the destination Pauli. It does so by

        1) converting any \|i+⟩ or \|i+⟩ eigenvector bits in the origin to
           \|+⟩ and \|-⟩ with S†s, then

        2) converting any \|+⟩ or \|+⟩ eigenvector bits in the converted origin to
           \|0⟩ and \|1⟩ with Hs, then

        3) writing the parity of the significant (Z-measured, rather than I)
           bits in the origin to a single
           "origin anchor bit," using cnots, which will hold the parity of these bits,

        4) swapping the parity of the pauli anchor bit into a destination anchor bit using
           a swap gate (only if they are different, if there are any bits which are significant
           in both origin and dest, we set both anchors to one of these bits to avoid a swap).

        5) writing the parity of the destination anchor bit into the other significant bits
           of the destination,

        6) converting the \|0⟩ and \|1⟩ significant eigenvector bits to \|+⟩ and \|-⟩ eigenvector
           bits in the destination where the destination demands it
           (e.g. pauli.x == true for a bit), using Hs 8) converting the \|+⟩ and \|-⟩
           significant eigenvector bits to \|i+⟩ and \|i-⟩ eigenvector bits in the
           destination where the destination demands it
           (e.g. pauli.x == true and pauli.z == true for a bit), using Ss

        Args:
            origin: The ``Pauli`` or ``PauliOp`` to map.

        Returns:
            A tuple of a ``PrimitiveOp`` which equals the basis change mapping and a ``PauliOp``
            which equals the destination basis.

        Raises:
            TypeError: Attempting to convert from non-Pauli origin.
            ValueError: Attempting to change a non-identity Pauli to an identity Pauli, or vice
                versa.

        """

        # If pauli is an PrimitiveOp, extract the Pauli
        if isinstance(origin, Pauli):
            origin = PauliOp(origin)

        if not isinstance(origin, PauliOp):
            raise TypeError(
                f"PauliBasisChange can only convert Pauli-based OpPrimitives, not {type(origin)}"
            )

        # If no destination specified, assume nearest Pauli in {Z,I}^n basis,
        # the standard basis change for expectations.
        destination = self.destination or self.get_diagonal_pauli_op(origin)

        # Pad origin or destination if either are not as long as the other
        origin, destination = self.pad_paulis_to_equal_length(origin, destination)

        origin_sig_bits = np.logical_or(origin.primitive.x, origin.primitive.z)
        destination_sig_bits = np.logical_or(destination.primitive.x, destination.primitive.z)
        if not any(origin_sig_bits) or not any(destination_sig_bits):
            if not (any(origin_sig_bits) or any(destination_sig_bits)):
                # Both all Identity, just return Identities
                return I ^ origin.num_qubits, destination
            else:
                # One is Identity, one is not
                raise ValueError("Cannot change to or from a fully Identity Pauli.")

        # Steps 1 and 2
        cob_instruction = self.get_diagonalizing_clifford(origin)

        # Construct CNOT chain, assuming full connectivity... - Steps 3)-5)
        cob_instruction = self.construct_cnot_chain(origin, destination).compose(cob_instruction)

        # Step 6 and 7
        dest_diagonlizing_clifford = self.get_diagonalizing_clifford(destination).adjoint()
        cob_instruction = dest_diagonlizing_clifford.compose(cob_instruction)

        return cast(PrimitiveOp, cob_instruction), destination
