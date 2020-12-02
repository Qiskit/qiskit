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

"""CVaRMeasurement class."""


from typing import Union, Optional, Callable, Tuple
import numpy as np

from qiskit.aqua import AquaError
from qiskit.circuit import ParameterExpression, QuantumCircuit, Instruction
from qiskit.result import Result
from qiskit.quantum_info import Statevector

from ..operator_base import OperatorBase
from ..list_ops import ListOp, SummedOp
from .state_fn import StateFn
from .operator_state_fn import OperatorStateFn


class CVaRMeasurement(OperatorStateFn):
    r"""A specialized measurement class to compute CVaR expectation values.
        See https://arxiv.org/pdf/1907.04769.pdf for further details.

    Used in :class:`~qiskit.aqua.operators.CVaRExpectation`, see there for more details.
    """
    def __new__(cls,
                primitive: Union[str, dict, Result,
                                 list, np.ndarray, Statevector,
                                 QuantumCircuit, Instruction,
                                 OperatorBase] = None,
                alpha: float = 1.0,
                coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                ) -> 'CVaRMeasurement':
        obj = object.__new__(cls)
        obj.__init__(primitive, alpha, coeff)
        return obj

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 alpha: float = 1.0,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> None:
        """
        Args:
            primitive: The ``OperatorBase`` which defines the diagonal operator
                       measurement.
            coeff: A coefficient by which to multiply the state function
            alpha: A real-valued parameter between 0 and 1 which specifies the
                   fraction of observed samples to include when computing the
                   objective value. alpha = 1 corresponds to a standard observable
                   expectation value. alpha = 0 corresponds to only using the single
                   sample with the lowest energy. alpha = 0.5 corresponds to ranking each
                   observation by lowest energy and using the best

        Raises:
            ValueError: TODO remove that this raises an error
            ValueError: If alpha is not in [0, 1].
            AquaError: If the primitive is not diagonal.
        """
        if primitive is None:
            raise ValueError

        if not 0 <= alpha <= 1:
            raise ValueError('The parameter alpha must be in [0, 1].')
        self._alpha = alpha

        if not _check_is_diagonal(primitive):
            raise AquaError('Input operator to CVaRMeasurement must be diagonal, but is not:',
                            str(primitive))

        super().__init__(primitive, coeff=coeff, is_measurement=True)

    @property
    def alpha(self) -> float:
        """A real-valued parameter between 0 and 1 which specifies the
           fraction of observed samples to include when computing the
           objective value. alpha = 1 corresponds to a standard observable
           expectation value. alpha = 0 corresponds to only using the single
           sample with the lowest energy. alpha = 0.5 corresponds to ranking each
           observation by lowest energy and using the best half.

        Returns:
            The parameter alpha which was given at initialization
        """
        return self._alpha

    def add(self, other: OperatorBase) -> OperatorBase:
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        """The adjoint of a CVaRMeasurement is not defined.

        Returns:
            Does not return anything, raises an error.

        Raises:
            AquaError: The adjoint of a CVaRMeasurement is not defined.
        """
        raise AquaError('Adjoint of a CVaR measurement not defined')

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              coeff=self.coeff * scalar,
                              alpha=self._alpha)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, OperatorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """Not defined."""
        raise NotImplementedError

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """Not defined."""
        raise NotImplementedError

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        """Not defined."""
        raise NotImplementedError

    def to_circuit_op(self) -> OperatorBase:
        """Not defined."""
        raise NotImplementedError

    def __str__(self) -> str:
        return 'CVaRMeasurement({}) * {}'.format(str(self.primitive), self.coeff)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[float, complex]:
        r"""
        Given the energies of each sampled measurement outcome (H_i) as well as the
        sampling probability of each measurement outcome (p_i, we can compute the
        CVaR as H_j + 1/α*(sum_i<j p_i*(H_i - H_j)). Note that index j corresponds
        to the measurement outcome such that only some of the samples with
        measurement outcome j will be used in computing CVaR. Note also that the
        sampling probabilities serve as an alternative to knowing the counts of each
        observation.

        This computation is broken up into two subroutines. One which evaluates each
        measurement outcome and determines the sampling probabilities of each. And one
        which carries out the above calculation. The computation is split up this way
        to enable a straightforward calculation of the variance of this estimator.

        Args:
            front: A StateFn or primitive which specifies the results of evaluating
                      a quantum state.

        Returns:
            The CVaR of the diagonal observable specified by self.primitive and
                the sampled quantum state described by the inputs
                (energies, probabilities). For index j (described above), the CVaR
                is computed as H_j + 1/α*(sum_i<j p_i*(H_i - H_j))
        """
        energies, probabilities = self.get_outcome_energies_probabilities(front)
        return self.compute_cvar(energies, probabilities)

    def eval_variance(self,
                      front: Union[str, dict, np.ndarray,
                                   OperatorBase] = None) -> Union[float, complex]:
        r"""
        Given the energies of each sampled measurement outcome (H_i) as well as the
        sampling probability of each measurement outcome (p_i, we can compute the
        variance of the CVaR estimator as
        H_j^2 + 1/α * (sum_i<j p_i*(H_i^2 - H_j^2)).
        This follows from the definition that Var[X] = E[X^2] - E[X]^2.
        In this case, X = E[<bi|H|bi>], where H is the diagonal observable and bi
        corresponds to measurement outcome i. Given this, E[X^2] = E[<bi|H|bi>^2]

        Args:
            front: A StateFn or primitive which specifies the results of evaluating
                      a quantum state.

        Returns:
            The Var[CVaR] of the diagonal observable specified by self.primitive
                and the sampled quantum state described by the inputs
                (energies, probabilities). For index j (described above), the CVaR
                is computed as H_j^2 + 1/α*(sum_i<j p_i*(H_i^2 - H_j^2))
        """
        energies, probabilities = self.get_outcome_energies_probabilities(front)
        sq_energies = [energy**2 for energy in energies]
        return self.compute_cvar(sq_energies, probabilities) - self.eval(front)**2

    def get_outcome_energies_probabilities(self,
                                           front: Union[str, dict, np.ndarray,
                                                        OperatorBase] = None) -> Tuple[list, list]:
        r"""
        In order to compute the  CVaR of an observable expectation, we require
        the energies of each sampled measurement outcome as well as the sampling
        probability of each measurement outcome. Note that the counts for each
        measurement outcome will also suffice (and this is often how the CVaR
        is presented).

        Args:
            front: A StateFn or a primitive which defines a StateFn.
                   This input holds the results of a sampled/simulated circuit.

        Returns:
            Two lists of equal length. `energies` contains the energy of each
                unique measurement outcome computed against the diagonal observable
                stored in self.primitive. `probabilities` contains the corresponding
                sampling probability for each measurement outcome in `energies`.

        Raises:
            ValueError: front isn't a DictStateFn or VectorStateFn
        """
        from .dict_state_fn import DictStateFn
        from .vector_state_fn import VectorStateFn
        from .circuit_state_fn import CircuitStateFn

        if isinstance(front, CircuitStateFn):
            front = front.eval()

        # Standardize the inputs to a dict
        if isinstance(front, DictStateFn):
            data = front.primitive
        elif isinstance(front, VectorStateFn):
            vec = front.primitive.data
            # Determine how many bits are needed
            key_len = int(np.ceil(np.log2(len(vec))))
            # Convert the vector primitive into a dict. The formatting here ensures
            # that the proper number of leading `0` characters are added.
            data = {format(index, '0'+str(key_len)+'b'): val for index, val in enumerate(vec)}
        else:
            raise ValueError('Unsupported input to CVaRMeasurement.eval:', type(front))

        obs = self.primitive
        outcomes = list(data.items())
        # add energy evaluation
        for i, outcome in enumerate(outcomes):
            key = outcome[0]
            outcomes[i] += (obs.eval(key).adjoint().eval(key),)

        # Sort each observation based on it's energy
        outcomes = sorted(outcomes, key=lambda x: x[2])

        # Here probabilities are the (root) probabilities of
        # observing each state. energies are the expectation
        # values of each state with the provided Hamiltonian.
        _, root_probabilities, energies = zip(*outcomes)

        # Square the dict values
        # (since CircuitSampler takes the root...)
        probabilities = [p_i * np.conj(p_i) for p_i in root_probabilities]
        return energies, probabilities

    def compute_cvar(self,
                     energies: list,
                     probabilities: list) -> Union[float, complex]:
        r"""
        Given the energies of each sampled measurement outcome (H_i) as well as the
        sampling probability of each measurement outcome (p_i, we can compute the
        CVaR. Note that the sampling probabilities serve as an alternative to knowing
        the counts of each observation and that the input energies are assumed to be
        sorted in increasing order.

        Consider the outcome with index j, such that only some of the samples with
        measurement outcome j will be used in computing CVaR. The CVaR calculation
        can then be separated into two parts. First we sum each of the energies for
        outcomes i < j, weighted by the probability of observing that outcome (i.e
        the normalized counts). Second, we add the energy for outcome j, weighted by
        the difference (α  - \sum_i<j p_i)

        Args:
            energies: A list containing the energies (H_i) of each sample measurement
                      outcome, sorted in increasing order.
            probabilities: The sampling probabilities (p_i) for each corresponding
                           measurement outcome.

        Returns:
            The CVaR of the diagonal observable specified by self.primitive and
                the sampled quantum state described by the inputs
                (energies, probabilities). For index j (described above), the CVaR
                is computed as H_j + 1/α * (sum_i<j p_i*(H_i - H_j))

        Raises:
            ValueError: front isn't a DictStateFn or VectorStateFn
        """
        alpha = self._alpha

        # Determine j, the index of the measurement outcome such
        # that only some samples with this outcome will be used to
        # compute the CVaR.
        j = 0
        running_total = 0
        for i, p_i in enumerate(probabilities):
            running_total += p_i
            j = i
            if running_total > alpha:
                break

        h_j = energies[j]
        cvar = alpha * h_j

        if alpha == 0 or j == 0:
            return self.coeff * h_j

        energies = energies[:j]
        probabilities = probabilities[:j]
        # Let H_i be the energy associated with outcome i
        # and let the outcomes be sorted by ascending energy.
        # Let p_i be the probability of observing outcome i.
        # CVaR = H_j + 1/α*(sum_i<j p_i*(H_i - H_j))
        for h_i, p_i in zip(energies, probabilities):
            cvar += p_i * (h_i - h_j)

        return self.coeff * cvar/alpha

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = None
                 ) -> OperatorBase:
        r"""
        Apply the convert_fn to the internal primitive if the primitive is an Operator (as in
        the case of ``OperatorStateFn``). Otherwise do nothing. Used by converters.

        Args:
            convert_fn: The function to apply to the internal OperatorBase.
            coeff: A coefficient to multiply by after applying convert_fn.
                If it is None, self.coeff is used instead.

        Returns:
            The converted StateFn.
        """
        if coeff is None:
            coeff = self.coeff

        if isinstance(self.primitive, OperatorBase):
            return self.__class__(convert_fn(self.primitive), coeff=coeff, alpha=self._alpha)
        return self

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        raise NotImplementedError


def _check_is_diagonal(operator: OperatorBase) -> bool:
    """Check whether ``operator`` is diagonal.

    Args:
        operator: The operator to check for diagonality.

    Returns:
        True, if the operator is diagonal, False otherwise.

    Raises:
        AquaError: If the operator is not diagonal.
    """
    # this must be a local import to avoid a cyclic import with components.uncertainty_models
    # TODO figure out why, it was not really clear to me why
    from ..primitive_ops import PauliOp
    if isinstance(operator, PauliOp):
        # every X component must be False
        if not np.any(operator.primitive.x):  # type: ignore
            return True
        return False

    if isinstance(operator, SummedOp) and operator.primitive_strings == {'Pauli'}:
        # cover the case of sums of diagonal paulis, but don't raise since there might be summands
        # canceling the non-diagonal parts

        # ignoring mypy since we know that all operators are PauliOps
        if np.all(not np.any(op.primitive.x) for op in operator.oplist):  # type: ignore
            return True

    if isinstance(operator, ListOp):
        return np.all(operator.traverse(_check_is_diagonal))

    # cannot efficiently check if a operator is diagonal, converting to matrix
    matrix = operator.to_matrix()

    if np.all(matrix == np.diag(np.diagonal(matrix))):
        return True
    return False
