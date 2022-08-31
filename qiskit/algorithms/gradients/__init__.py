# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
==============================================
Gradients (:mod:`qiskit.algorithms.gradients`)
==============================================

.. currentmodule:: qiskit.algorithms.gradients

Base Classes
============

.. autosummary::
   :toctree: ../stubs/

   BaseSamplerGradient
   BaseEstimatorGradient

Estimator Gradients
===================

.. autosummary::
   :toctree: ../stubs/

   FiniteDiffEstimatorGradient
   LinCombEstimatorGradient
   ParamShiftEstimatorGradient
   SPSAEstimatorGradient

Sampler Gradients
=================

.. autosummary::
   :toctree: ../stubs/

   FiniteDiffSamplerGradient
   LinCombSamplerGradient
   ParamShiftSamplerGradient
   SPSASamplerGradient

Results
=======

.. autosummary::
   :toctree: ../stubs/

   EstimatorGradientResult
   SamplerGradientResult
"""

from .base_estimator_gradient import BaseEstimatorGradient
from .base_sampler_gradient import BaseSamplerGradient
from .estimator_gradient_result import EstimatorGradientResult
from .finite_diff_estimator_gradient import FiniteDiffEstimatorGradient
from .finite_diff_sampler_gradient import FiniteDiffSamplerGradient
from .lin_comb_estimator_gradient import LinCombEstimatorGradient
from .lin_comb_sampler_gradient import LinCombSamplerGradient
from .param_shift_estimator_gradient import ParamShiftEstimatorGradient
from .param_shift_sampler_gradient import ParamShiftSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .spsa_estimator_gradient import SPSAEstimatorGradient
from .spsa_sampler_gradient import SPSASamplerGradient

__all__ = [
    "BaseEstimatorGradient",
    "BaseSamplerGradient",
    "EstimatorGradientResult",
    "FiniteDiffEstimatorGradient",
    "FiniteDiffSamplerGradient",
    "LinCombEstimatorGradient",
    "LinCombSamplerGradient",
    "ParamShiftEstimatorGradient",
    "ParamShiftSamplerGradient",
    "SamplerGradientResult",
    "SPSAEstimatorGradient",
    "SPSASamplerGradient",
]
