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
Abstract base classes for primitives module.
"""

from .base_sampler import BaseSampler, BaseSamplerV2
from .base_estimator import BaseEstimator, BaseEstimatorV2
from .estimator_result import EstimatorResult
from .sampler_result import SamplerResult
