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
=====================================
Primitives (:mod:`qiskit.primitives`)
=====================================

.. currentmodule:: qiskit.primitives

.. automodule:: qiskit.primitives.base_estimator
.. automodule:: qiskit.primitives.base_sampler

.. currentmodule:: qiskit.primitives

Estimator
=========

.. autosummary::
   :toctree: ../stubs/

   BaseEstimator
   Estimator

Sampler
=======

.. autosummary::
   :toctree: ../stubs/

   BaseSampler
   Sampler

Results
=======

.. autosummary::
   :toctree: ../stubs/

   EstimatorResult
   SamplerResult
"""

from .base_estimator import BaseEstimator
from .base_sampler import BaseSampler
from .estimator import Estimator
from .estimator_result import EstimatorResult
from .sampler import Sampler
from .sampler_result import SamplerResult
