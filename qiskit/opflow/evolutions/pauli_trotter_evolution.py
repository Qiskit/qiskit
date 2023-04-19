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

"""PauliTrotterEvolution Class"""

import logging
from typing import Optional, Union, cast

import numpy as np

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.opflow.converters.pauli_basis_change import PauliBasisChange
from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow.evolutions.evolved_op import EvolvedOp
from qiskit.opflow.evolutions.trotterizations.trotterization_base import TrotterizationBase
from qiskit.opflow.evolutions.trotterizations.trotterization_factory import TrotterizationFactory
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.operator_globals import I, Z
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.utils.deprecation import deprecate_func

# TODO uncomment when we implement Abelian grouped evolution.
# from qiskit.opflow.converters.abelian_grouper import AbelianGrouper

logger = logging.getLogger(__name__)


class PauliTrotterEvolution(EvolutionBase):
    r"""
    Deprecated: An Evolution algorithm replacing exponentiated sums of Paulis by changing
    them each to the Z basis, rotating with an rZ, changing back, and Trotterizing.

    More specifically, we compute basis change circuits for each Pauli into a single-qubit Z,
    evolve the Z by the desired evolution time with an rZ gate, and change the basis back using
    the adjoint of the original basis change circuit. For sums of Paulis, the individual Pauli
    evolution circuits are composed together by Trotterization scheme.
    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        trotter_mode: Optional[Union[str, TrotterizationBase]] = "trotter",
        reps: Optional[int] = 1,
        # TODO uncomment when we implement Abelian grouped evolution.
        # group_paulis: Optional[bool] = False
    ) -> None:
        """
        Args:
            trotter_mode: A string ('trotter', 'suzuki', or 'qdrift') to pass to the
                TrotterizationFactory, or a TrotterizationBase, indicating how to combine
                individual Pauli evolution circuits to equal the exponentiation of the Pauli sum.
            reps: How many Trotterization repetitions to make, to improve the approximation
                accuracy.
            # TODO uncomment when we implement Abelian grouped evolution.
            # group_paulis: Whether to group Pauli sums into Abelian
            #     sub-groups, so a single diagonalization circuit can be used for each group
            #     rather than each Pauli.
        """
        super().__init__()
        if isinstance(trotter_mode, TrotterizationBase):
            self._trotter = trotter_mode
        else:
            self._trotter = TrotterizationFactory.build(mode=trotter_mode, reps=reps)

        # TODO uncomment when we implement Abelian grouped evolution.
        # self._grouper = AbelianGrouper() if group_paulis else None

    @property
    def trotter(self) -> TrotterizationBase:
        """TrotterizationBase used to evolve SummedOps."""
        return self._trotter

    @trotter.setter
    def trotter(self, trotter: TrotterizationBase) -> None:
        """Set TrotterizationBase used to evolve SummedOps."""
        self._trotter = trotter

    def convert(self, operator: OperatorBase) -> OperatorBase:
        r"""
        Traverse the operator, replacing ``EvolvedOps`` with ``CircuitOps`` containing
        Trotterized evolutions equalling the exponentiation of -i * operator.

        Args:
            operator: The Operator to convert.

        Returns:
            The converted operator.
        """
        # TODO uncomment when we implement Abelian grouped evolution.
        # if self._grouper:
        #     # Sort into commuting groups
        #     operator = self._grouper.convert(operator).reduce()
        return self._recursive_convert(operator)

    def _get_evolution_synthesis(self):
        """Return the ``EvolutionSynthesis`` corresponding to this Trotterization."""
        if self.trotter.order == 1:
            return LieTrotter(reps=self.trotter.reps)
        return SuzukiTrotter(reps=self.trotter.reps, order=self.trotter.order)

    def _recursive_convert(self, operator: OperatorBase) -> OperatorBase:
        if isinstance(operator, EvolvedOp):
            if isinstance(operator.primitive, (PauliOp, PauliSumOp)):
                pauli = operator.primitive.primitive
                time = operator.coeff * operator.primitive.coeff
                evo = PauliEvolutionGate(
                    pauli, time=time, synthesis=self._get_evolution_synthesis()
                )
                return CircuitOp(evo)
                # operator = EvolvedOp(operator.primitive.to_pauli_op(), coeff=operator.coeff)
            if not {"Pauli"} == operator.primitive_strings():
                logger.warning(
                    "Evolved Hamiltonian is not composed of only Paulis, converting to "
                    "Pauli representation, which can be expensive."
                )
                # Setting massive=False because this conversion is implicit. User can perform this
                # action on the Hamiltonian with massive=True explicitly if they so choose.
                # TODO explore performance to see whether we should avoid doing this repeatedly
                pauli_ham = operator.primitive.to_pauli_op(massive=False)
                operator = EvolvedOp(pauli_ham, coeff=operator.coeff)

            if isinstance(operator.primitive, SummedOp):
                # TODO uncomment when we implement Abelian grouped evolution.
                # if operator.primitive.abelian:
                #     return self.evolution_for_abelian_paulisum(operator.primitive)
                # else:
                # Collect terms that are not the identity.
                oplist = [
                    x
                    for x in operator.primitive
                    if not isinstance(x, PauliOp) or sum(x.primitive.x + x.primitive.z) != 0
                ]
                # Collect the coefficients of any identity terms,
                # which become global phases when exponentiated.
                identity_phases = [
                    x.coeff
                    for x in operator.primitive
                    if isinstance(x, PauliOp) and sum(x.primitive.x + x.primitive.z) == 0
                ]
                # Construct sum without the identity operators.
                new_primitive = SummedOp(oplist, coeff=operator.primitive.coeff)
                trotterized = self.trotter.convert(new_primitive)
                circuit_no_identities = self._recursive_convert(trotterized)
                # Set the global phase of the QuantumCircuit to account for removed identity terms.
                global_phase = -sum(identity_phases) * operator.primitive.coeff
                circuit_no_identities.primitive.global_phase = global_phase
                return circuit_no_identities
            # Covers ListOp, ComposedOp, TensoredOp
            elif isinstance(operator.primitive, ListOp):
                converted_ops = [self._recursive_convert(op) for op in operator.primitive.oplist]
                return operator.primitive.__class__(converted_ops, coeff=operator.coeff)
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert).reduce()

        return operator

    def evolution_for_pauli(self, pauli_op: PauliOp) -> PrimitiveOp:
        r"""
        Compute evolution Operator for a single Pauli using a ``PauliBasisChange``.

        Args:
            pauli_op: The ``PauliOp`` to evolve.

        Returns:
            A ``PrimitiveOp``, either the evolution ``CircuitOp`` or a ``PauliOp`` equal to the
            identity if pauli_op is the identity.
        """

        def replacement_fn(cob_instr_op, dest_pauli_op):
            z_evolution = dest_pauli_op.exp_i()
            # Remember, circuit composition order is mirrored operator composition order.
            return cob_instr_op.adjoint().compose(z_evolution).compose(cob_instr_op)

        # Note: PauliBasisChange will pad destination with identities
        # to produce correct CoB circuit
        sig_bits = np.logical_or(pauli_op.primitive.z, pauli_op.primitive.x)
        a_sig_bit = int(max(np.extract(sig_bits, np.arange(pauli_op.num_qubits)[::-1])))
        destination = (I.tensorpower(a_sig_bit)) ^ (Z * pauli_op.coeff)
        cob = PauliBasisChange(destination_basis=destination, replacement_fn=replacement_fn)
        return cast(PrimitiveOp, cob.convert(pauli_op))

    # TODO implement Abelian grouped evolution.
    def evolution_for_abelian_paulisum(self, op_sum: SummedOp) -> PrimitiveOp:
        """Evolution for abelian pauli sum"""
        raise NotImplementedError
