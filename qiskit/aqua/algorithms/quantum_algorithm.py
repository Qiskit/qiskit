# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for algorithm modules.

To create add-on algorithm modules subclass the QuantumAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import abstractmethod
import logging
from qiskit.aqua import aqua_globals, Pluggable, QuantumInstance, AquaError

logger = logging.getLogger(__name__)


class QuantumAlgorithm(Pluggable):
    """
    Base class for Algorithms.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.
    """
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._quantum_instance = None

    @property
    def random(self):
        """Return a numpy random."""
        return aqua_globals.random

    def run(self, quantum_instance=None, **kwargs):
        """Execute the algorithm with selected backend.

        Args:
            quantum_instance (QuantumInstance or BaseBackend): the experimental setting.
            kwargs (dict): kwargs
        Returns:
            dict: results of an algorithm.
        """
        # pylint: disable=import-outside-toplevel
        from qiskit.providers import BaseBackend

        if not self.configuration.get('classical', False):
            if quantum_instance is None:
                AquaError("Quantum device or backend "
                          "is needed since you are running quantum algorithm.")
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
        """ returns quantum instance """
        return self._quantum_instance
