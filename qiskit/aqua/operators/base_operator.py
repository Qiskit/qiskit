# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC, abstractmethod


class BaseOperator(ABC):
    """Operators relevant for quantum applications."""

    @abstractmethod
    def __init__(self):
        """Constructor."""
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @abstractmethod
    def __add__(self, other):
        """Overload + operation."""
        raise NotImplementedError

    @abstractmethod
    def __iadd__(self, other):
        """Overload += operation."""
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other):
        """Overload - operation."""
        raise NotImplementedError

    @abstractmethod
    def __isub__(self, other):
        """Overload -= operation."""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        """Overload unary - ."""
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        """Overload == operation."""
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        """Overload str()."""
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other):
        """Overload *."""
        raise NotImplementedError

    @abstractmethod
    def construct_evaluation_circuit(self, wave_function):
        """Build circuits to compute the expectation w.r.t the wavefunction."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_with_result(self, result):
        """
        Consume the result from the quantum computer to build the expectation,
        will be only used along with the `construct_evaluation_circuit` method.
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self):
        """
        Time evolution, exp^(-jt H).
        """
        raise NotImplementedError
