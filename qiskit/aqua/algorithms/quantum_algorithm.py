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

"""
This module implements the abstract base class for algorithm modules.

To create add-on algorithm modules subclass the QuantumAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
from qiskit.providers import BaseBackend
from qiskit.providers.backend import Backend
from qiskit.aqua import aqua_globals, QuantumInstance, AquaError


class QuantumAlgorithm(ABC):
    """
    Base class for Quantum Algorithms.

    This method should initialize the module and
    use an exception if a component of the module is available.
    """
    @abstractmethod
    def __init__(self,
                 quantum_instance: Optional[
                     Union[QuantumInstance, Backend, BaseBackend, Backend]]) -> None:
        self._quantum_instance = None
        if quantum_instance:
            self.quantum_instance = quantum_instance

    @property
    def random(self):
        """Return a numpy random."""
        return aqua_globals.random

    def run(self,
            quantum_instance: Optional[
                Union[QuantumInstance, Backend, BaseBackend]] = None,
            **kwargs) -> Dict:
        """Execute the algorithm with selected backend.

        Args:
            quantum_instance: the experimental setting.
            kwargs (dict): kwargs
        Returns:
            dict: results of an algorithm.
        Raises:
            AquaError: If a quantum instance or backend has not been provided
        """
        if quantum_instance is None and self.quantum_instance is None:
            raise AquaError("A QuantumInstance or Backend "
                            "must be supplied to run the quantum algorithm.")
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            self.set_backend(quantum_instance, **kwargs)
        else:
            if quantum_instance is not None:
                self.quantum_instance = quantum_instance

        return self._run()

    @abstractmethod
    def _run(self) -> Dict:
        raise NotImplementedError()

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """ Returns quantum instance. """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance,
                                                       BaseBackend, Backend]) -> None:
        """ Sets quantum instance. """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def set_backend(self, backend: Union[Backend, BaseBackend], **kwargs) -> None:
        """ Sets backend with configuration. """
        self.quantum_instance = QuantumInstance(backend)
        self.quantum_instance.set_config(**kwargs)

    @property
    def backend(self) -> Union[Backend, BaseBackend]:
        """ Returns backend. """
        return self.quantum_instance.backend

    @backend.setter
    def backend(self, backend: Union[Backend, BaseBackend]):
        """ Sets backend without additional configuration. """
        self.set_backend(backend)
