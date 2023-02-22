# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023
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


Estimator Gradients
===================

.. autosummary::
   :toctree: ../stubs/

   BaseEstimatorGradient
   DerivativeType
   FiniteDiffEstimatorGradient
   LinCombEstimatorGradient
   ParamShiftEstimatorGradient
   SPSAEstimatorGradient
   ReverseEstimatorGradient

Sampler Gradients
=================

.. autosummary::
   :toctree: ../stubs/

   BaseSamplerGradient
   FiniteDiffSamplerGradient
   LinCombSamplerGradient
   ParamShiftSamplerGradient
   SPSASamplerGradient

Quantum Geometric Tensor
========================
.. autosummary::
   :toctree: ../stubs/

   BaseQGT
   LinCombQGT
   QFI
   ReverseQGT

Results
=======

.. autosummary::
   :toctree: ../stubs/

   EstimatorGradientResult
   QFIResult
   QGTResult
   SamplerGradientResult
"""

from .base_estimator_gradient import BaseEstimatorGradient
from .base_qgt import BaseQGT
from .base_sampler_gradient import BaseSamplerGradient
from .estimator_gradient_result import EstimatorGradientResult
from .finite_diff_estimator_gradient import FiniteDiffEstimatorGradient
from .finite_diff_sampler_gradient import FiniteDiffSamplerGradient
from .lin_comb_estimator_gradient import DerivativeType, LinCombEstimatorGradient
from .lin_comb_qgt import LinCombQGT
from .lin_comb_sampler_gradient import LinCombSamplerGradient
from .param_shift_estimator_gradient import ParamShiftEstimatorGradient
from .param_shift_sampler_gradient import ParamShiftSamplerGradient
from .qfi import QFI
from .qfi_result import QFIResult
from .qgt_result import QGTResult
from .sampler_gradient_result import SamplerGradientResult
from .spsa_estimator_gradient import SPSAEstimatorGradient
from .spsa_sampler_gradient import SPSASamplerGradient
from .reverse_gradient.reverse_gradient import ReverseEstimatorGradient
from .reverse_gradient.reverse_qgt import ReverseQGT

__all__ = [
    "BaseEstimatorGradient",
    "BaseQGT",
    "BaseSamplerGradient",
    "DerivativeType",
    "EstimatorGradientResult",
    "FiniteDiffEstimatorGradient",
    "FiniteDiffSamplerGradient",
    "LinCombEstimatorGradient",
    "LinCombQGT",
    "LinCombSamplerGradient",
    "ParamShiftEstimatorGradient",
    "ParamShiftSamplerGradient",
    "QFI",
    "QFIResult",
    "QGTResult",
    "SamplerGradientResult",
    "SPSAEstimatorGradient",
    "SPSASamplerGradient",
    "ReverseEstimatorGradient",
    "ReverseQGT",
]
