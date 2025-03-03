# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analysis pass to find commutation relations between DAG nodes."""

from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit._accelerate.commutation_analysis import analyze_commutations


class CommutationAnalysis(AnalysisPass):
    r"""Analysis pass to find commutation relations between DAG nodes.

    This sets ``property_set['commutation_set']`` to a dictionary that describes
    the commutation relations on a given wire: all the gates on a wire
    are grouped into a set of gates that commute.

    When possible, commutation relations are queried from a lookup table. This is the case
    for standard gates without parameters (such as :class:`.XGate` or :class:`.HGate`) or
    gates with free parameters (such as :class:`.RXGate` with a :class:`.ParameterExpression` as
    angle). Otherwise, a matrix-based check is performed, where two operations are said to
    commute, if the average gate fidelity of performing the commutation is above a certain threshold.
    Concretely, two unitaries :math:`A` and :math:`B` on :math:`n` qubits commute if

    .. math::

        \frac{2^n F_{\text{process}}(AB, BA) + 1}{2^n + 1} > 1 - \varepsilon,

    where

    .. math::

        F_{\text{process}}(U_1, U_2) = \left|\frac{\mathrm{Tr}(U_1 U_2^\dagger)}{2^n} \right|^2,

    and we set :math:`\varepsilon` to ``16 * machine_eps`` to account for round-off errors on
    few-qubit systems.
    """

    def __init__(self, *, _commutation_checker=None):
        super().__init__()
        # allow setting a private commutation checker, this allows better performance if we
        # do not care about commutations of all gates, but just a subset
        if _commutation_checker is None:
            _commutation_checker = scc

        self.comm_checker = _commutation_checker

    def run(self, dag):
        """Run the CommutationAnalysis pass on `dag`.

        Run the pass on the DAG, and write the discovered commutation relations
        into the ``property_set``.
        """
        # Initiate the commutation set
        self.property_set["commutation_set"] = analyze_commutations(dag, self.comm_checker.cc)
