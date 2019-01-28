# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
This module implements the abstract base class for algorithm modules.

To create add-on algorithm modules subclass the QuantumAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import abstractmethod
import logging

import numpy as np
from qiskit.providers import BaseBackend

from qiskit.aqua import Pluggable, QuantumInstance, AquaError

logger = logging.getLogger(__name__)


class QuantumAlgorithm(Pluggable):

    # Configuration dictionary keys
    SECTION_KEY_ALGORITHM = 'algorithm'
    SECTION_KEY_OPTIMIZER = 'optimizer'
    SECTION_KEY_VAR_FORM = 'variational_form'
    SECTION_KEY_INITIAL_STATE = 'initial_state'
    SECTION_KEY_IQFT = 'iqft'
    SECTION_KEY_ORACLE = 'oracle'
    SECTION_KEY_FEATURE_MAP = 'feature_map'
    SECTION_KEY_MULTICLASS_EXTENSION = 'multiclass_extension'
    SECTION_KEY_UNCERTAINTY_PROBLEM = 'uncertainty_problem'
    SECTION_KEY_UNCERTAINTY_MODEL = 'uncertainty_model'

    """
    Base class for Algorithms.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    Args:
        configuration (dict): configuration dictionary
    """
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._random_seed = None
        self._random = None
        self._quantum_instance = None

    @property
    def random_seed(self):
        """Return random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """Set random seed."""
        self._random_seed = seed

    @property
    def random(self):
        """Return a numpy random."""
        if self._random is None:
            if self._random_seed is None:
                self._random = np.random
            else:
                self._random = np.random.RandomState(self._random_seed)
        return self._random

    def run(self, quantum_instance=None, **kwargs):
        """Execute the algorithm with selected backend.

        Args:
            quantum_instance (QuantumInstance or BaseBackend): the experiemental setting.

        Returns:
            dict: results of an algorithm.
        """
        if not self.configuration.get('classical', False):
            if quantum_instance is None:
                AquaError("Quantum device or backend is needed since you are running quanutm algorithm.")
            if isinstance(quantum_instance, BaseBackend):
                quantum_instance = QuantumInstance(quantum_instance)
                quantum_instance.set_config(**kwargs)
            self._quantum_instance = quantum_instance
        return self._run()

    @abstractmethod
    def _run(self):
        raise NotImplementedError()

    @property
    def quantum_instance(self):
        return self._quantum_instance
