# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" The Quantum Approximate Optimization Algorithm. """

from typing import List, Callable, Optional, Union
import logging
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import OperatorBase, ExpectationBase, LegacyBaseOperator
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.utils.validation import validate_min
from .var_form import QAOAVarForm
from ..vqe import VQE

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name
# disable check for operator setter because of pylint bug
# pylint: disable=no-member


class QAOA(VQE):
    """
    The Quantum Approximate Optimization Algorithm.

    `QAOA <https://arxiv.org/abs/1411.4028>`__ is a well-known algorithm for finding approximate
    solutions to combinatorial-optimization problems.
    The QAOA implementation in Aqua directly extends :class:`VQE` and inherits VQE's
    general hybrid optimization structure.
    However, unlike VQE, which can be configured with arbitrary variational forms,
    QAOA uses its own fine-tuned variational form, which comprises :math:`p` parameterized global
    :math:`x` rotations and :math:`p` different parameterizations of the problem hamiltonian.
    QAOA is thus principally configured  by the single integer parameter, *p*,
    which dictates the depth of the variational form, and thus affects the approximation quality.

    An optional array of :math:`2p` parameter values, as the *initial_point*, may be provided as the
    starting **beta** and **gamma** parameters (as identically named in the
    original `QAOA paper <https://arxiv.org/abs/1411.4028>`__) for the QAOA variational form.

    An operator or a parameterized quantum circuit may optionally also be provided as a custom
    `mixer` Hamiltonian. This allows, as discussed in
    `this paper <https://doi.org/10.1103/PhysRevApplied.5.034007>`__ for quantum annealing,
    and in `this paper <https://arxiv.org/abs/1709.03489>`__ for QAOA,
    to run constrained optimization problems where the mixer constrains
    the evolution to a feasible subspace of the full Hilbert space.

    An initial state from Aqua's :mod:`~qiskit.aqua.components.initial_states` may optionally
    be supplied.
    """

    def __init__(self,
                 operator: Union[OperatorBase, LegacyBaseOperator] = None,
                 optimizer: Optimizer = None,
                 p: int = 1,
                 initial_state: Optional[Union[QuantumCircuit, InitialState]] = None,
                 mixer: Union[QuantumCircuit, OperatorBase, LegacyBaseOperator] = None,
                 initial_point: Optional[np.ndarray] = None,
                 gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]],
                                                                 List]]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False,
                 max_evals_grouped: int = 1,
                 aux_operators: Optional[List[Optional[Union[OperatorBase, LegacyBaseOperator]]]] =
                 None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """
        Args:
            operator: Qubit operator
            optimizer: A classical optimizer.
            p: the integer parameter p as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
                as well as warm-starting the optimization as introduced
                in http://arxiv.org/abs/2009.10095.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then it will simply compute a random one.
            gradient: An optional gradient operator respectively a gradient function used for
                      optimization.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When None (the default) an
                :class:`~qiskit.aqua.operators.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to True (defaults to False).
            include_custom: When `expectation` parameter here is None setting this to True will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Ignored if a gradient operator or function is
                given.
            aux_operators: Optional list of auxiliary operators to be evaluated with the eigenstate
                of the minimum eigenvalue main result and their expectation values returned.
                For instance in chemistry these can be dipole operators, total particle count
                operators so we can get values for these at the ground state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('p', p, 1)

        self._p = p
        self._mixer = mixer.to_opflow() if isinstance(mixer, LegacyBaseOperator) else mixer
        self._initial_state = initial_state

        # VQE will use the operator setter, during its constructor, which is overridden below and
        # will cause the var form to be built
        super().__init__(operator,
                         None,
                         optimizer,
                         initial_point=initial_point,
                         gradient=gradient,
                         expectation=expectation,
                         include_custom=include_custom,
                         max_evals_grouped=max_evals_grouped,
                         callback=callback,
                         quantum_instance=quantum_instance,
                         aux_operators=aux_operators)

    @VQE.operator.setter  # type: ignore
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        """ Sets operator """
        # Need to wipe the var_form in case number of qubits differs from operator.
        self.var_form = None
        # Setting with VQE's operator property
        super(QAOA, self.__class__).operator.__set__(self, operator)  # type: ignore
        self.var_form = QAOAVarForm(self.operator,
                                    self._p,
                                    initial_state=self._initial_state,
                                    mixer_operator=self._mixer)

    @property
    def initial_state(self) -> Optional[Union[QuantumCircuit, InitialState]]:
        """
        Returns:
            Returns the initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[Union[QuantumCircuit, InitialState]]) -> None:
        """
        Args:
            initial_state: Initial state to set.
        """
        self._initial_state = initial_state

    @property
    def mixer(self) -> Union[QuantumCircuit, OperatorBase, LegacyBaseOperator]:
        """
        Returns:
            Returns the mixer.
        """
        return self._mixer

    @mixer.setter
    def mixer(self, mixer: Union[QuantumCircuit, OperatorBase, LegacyBaseOperator]) -> None:
        """
        Args:
            mixer: Mixer to set.
        """
        self._mixer = mixer
