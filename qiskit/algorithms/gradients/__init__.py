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

Base Classes
============

.. autosummary::
   :toctree: ../stubs/

   BaseEstimatorGradient
   BaseQGT
   BaseSamplerGradient
   EstimatorGradientResult
   SamplerGradientResult
   QGTResult

Finite Differences
==================

.. autosummary::
   :toctree: ../stubs/

   FiniteDiffEstimatorGradient
   FiniteDiffSamplerGradient

Linear Combination of Unitaries
===============================

.. autosummary::
   :toctree: ../stubs/

   LinCombEstimatorGradient
   LinCombSamplerGradient
   LinCombQGT

Parameter Shift Rules
=====================

.. autosummary::
   :toctree: ../stubs/

   ParamShiftEstimatorGradient
   ParamShiftSamplerGradient

Quantum Fisher Information
==========================

.. autosummary::
   :toctree: ../stubs/

   QFIResult
   QFI

Classical Methods
=================

.. autosummary::
   :toctree: ../stubs/

   ReverseEstimatorGradient
   ReverseQGT

Simultaneous Perturbation Stochastic Approximation
==================================================

.. autosummary::
   :toctree: ../stubs/

   SPSAEstimatorGradient
   SPSASamplerGradient
"""

from .base.base_estimator_gradient import BaseEstimatorGradient
from .base.base_qgt import BaseQGT
from .base.base_sampler_gradient import BaseSamplerGradient
from .base.estimator_gradient_result import EstimatorGradientResult
from .finite_diff.finite_diff_estimator_gradient import FiniteDiffEstimatorGradient
from .finite_diff.finite_diff_sampler_gradient import FiniteDiffSamplerGradient
from .lin_comb.lin_comb_estimator_gradient import DerivativeType, LinCombEstimatorGradient
from .lin_comb.lin_comb_qgt import LinCombQGT
from .lin_comb.lin_comb_sampler_gradient import LinCombSamplerGradient
from .param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
from .param_shift.param_shift_sampler_gradient import ParamShiftSamplerGradient
from .qfi import QFI
from .qfi_result import QFIResult
from .base.qgt_result import QGTResult
from .base.sampler_gradient_result import SamplerGradientResult
from .spsa.spsa_estimator_gradient import SPSAEstimatorGradient
from .spsa.spsa_sampler_gradient import SPSASamplerGradient
from .reverse.reverse_gradient import ReverseEstimatorGradient
from .reverse.reverse_qgt import ReverseQGT

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
