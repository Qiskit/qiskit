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

"""Controlled Random Search (CRS) with local mutation."""

from .nloptimizer import NLoptOptimizer, NLoptOptimizerType


class CRS(NLoptOptimizer):
    # pylint: disable=line-too-long
    """
    Controlled Random Search (CRS) with local mutation.

    NLopt global optimizer, derivative-free. See `NLOpt CRS documentation
    <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#controlled-random-search-crs-with-local-mutation>`_
    for more information.
    """

    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """ return NLopt optimizer type """
        return NLoptOptimizerType.GN_CRS2_LM
