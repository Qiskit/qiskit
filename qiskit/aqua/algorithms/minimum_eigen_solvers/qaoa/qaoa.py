# -*- coding: utf-8 -*-

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

from typing import List, Callable, Optional
import logging
import numpy as np
from qiskit.aqua.operators import BaseOperator
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

    An operator may optionally also be provided as a custom `mixer` Hamiltonian. This allows,
    as discussed in `this paper <https://doi.org/10.1103/PhysRevApplied.5.034007>`__
    for quantum annealing, and in `this paper <https://arxiv.org/abs/1709.03489>`__ for QAOA,
    to run constrained optimization problems where the mixer constrains
    the evolution to a feasible subspace of the full Hilbert space.

    An initial state from Aqua's :mod:`~qiskit.aqua.components.initial_states` may optionally
    be supplied.
    """

    def __init__(self, operator: BaseOperator = None, optimizer: Optimizer = None, p: int = 1,
                 initial_state: Optional[InitialState] = None,
                 mixer: Optional[BaseOperator] = None, initial_point: Optional[np.ndarray] = None,
                 max_evals_grouped: int = 1, aux_operators: Optional[List[BaseOperator]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 auto_conversion: bool = True) -> None:
        """
        Args:
            operator: Qubit operator
            optimizer: A classical optimizer.
            p: the integer parameter p as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with. Allows support of optimizations in
                constrained subspaces as per https://arxiv.org/abs/1709.03489
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then it will simply compute a random one.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time.
            aux_operators: Optional list of auxiliary operators to be evaluated with the eigenstate
                of the minimum eigenvalue main result and their expectation values returned.
                For instance in chemistry these can be dipole operators, total particle count
                operators so we can get values for these at the ground state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.
            auto_conversion: When ``True`` allows an automatic conversion for operator and
                aux_operators into the type which is most suitable for the backend on which the
                algorithm is run.

                - for *non-Aer statevector simulator:*
                  :class:`~qiskit.aqua.operators.MatrixOperator`
                - for *Aer statevector simulator:*
                  :class:`~qiskit.aqua.operators.WeightedPauliOperator`
                - for *qasm simulator or real backend:*
                  :class:`~qiskit.aqua.operators.TPBGroupedWeightedPauliOperator`
        """
        validate_min('p', p, 1)

        self._p = p
        self._mixer_operator = mixer
        self._initial_state = initial_state

        # VQE will use the operator setter, during its constructor, which is overridden below and
        # will cause the var form to be built
        super().__init__(operator, None, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped, aux_operators=aux_operators,
                         callback=callback, auto_conversion=auto_conversion)

    @VQE.operator.setter
    def operator(self, operator: BaseOperator) -> None:
        """ Sets operator """
        if operator is not None:
            self._in_operator = operator
            self.var_form = QAOAVarForm(operator.copy(),
                                        self._p,
                                        initial_state=self._initial_state,
                                        mixer_operator=self._mixer_operator)
