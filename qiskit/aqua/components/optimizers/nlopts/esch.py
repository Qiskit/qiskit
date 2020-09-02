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

""" ESCH evolutionary optimizer. """

from .nloptimizer import NLoptOptimizer, NLoptOptimizerType


class ESCH(NLoptOptimizer):
    """
    ESCH evolutionary optimizer.

    ESCH is an evolutionary algorithm for global optimization that supports bound constraints only.
    Specifically, it does not support nonlinear constraints.

    NLopt global optimizer, derivative-free.
    For further detail, please refer to

    http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#esch-evolutionary-algorithm
    """

    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """ Return NLopt optimizer type """
        return NLoptOptimizerType.GN_ESCH
