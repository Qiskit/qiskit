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

.. automodule:: qiskit.primitives.base.base_estimator
.. automodule:: qiskit.primitives.base.base_sampler

.. currentmodule:: qiskit.primitives

Estimator
=========

.. autosummary::
   :toctree: ../stubs/

   BaseEstimator
   Estimator
   BackendEstimator

EstimatorV2
===========

.. autosummary::
   :toctree: ../stubs/

   StatevectorEstimator

Sampler
=======

.. autosummary::
   :toctree: ../stubs/

   BaseSampler
   Sampler
   BackendSampler
   BaseSamplerV2
   StatevectorSampler

Results
=======

.. autosummary::
   :toctree: ../stubs/

   EstimatorResult
   SamplerResult
   PrimitiveResult
   PubResult
"""

from .backend_estimator import BackendEstimator
from .backend_sampler import BackendSampler
from .base import BaseEstimator, BaseSampler, BaseSamplerV2
from .base.estimator_result import EstimatorResult
from .base.sampler_result import SamplerResult
from .containers import (
    PrimitiveResult,
    PubResult,
    EstimatorPubLike,
    SamplerPubLike,
    BindingsArrayLike,
    ObservableLike,
    ObservablesArrayLike,
)
from .estimator import Estimator
from .sampler import Sampler
from .statevector_estimator import StatevectorEstimator
from .statevector_sampler import StatevectorSampler
