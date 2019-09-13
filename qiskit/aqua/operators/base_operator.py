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

""" Base Operator """

from abc import ABC, abstractmethod


class BaseOperator(ABC):
    """Operators relevant for quantum applications."""

    @abstractmethod
    def __init__(self, basis=None, z2_symmetries=None, name=None):
        """Constructor."""
        self._basis = basis
        self._z2_symmetries = z2_symmetries
        self._name = name if name is not None else ''

    @property
    def name(self):
        """ returns name """
        return self._name

    @name.setter
    def name(self, new_value):
        """ sets name """
        self._name = new_value

    @property
    def basis(self):
        """ returns basis """
        return self._basis

    @property
    def z2_symmetries(self):
        """ returns z2 symmetries """
        return self._z2_symmetries

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
    def construct_evaluation_circuit(self, wave_function, statevector_mode, **kwargs):
        """Build circuits to compute the expectation w.r.t the wavefunction."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_with_result(self, result, statevector_mode, **kwargs):
        """
        Consume the result from the quantum computer to build the expectation,
        will be only used along with the `construct_evaluation_circuit` method.
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, state_in, evo_time, num_time_slices, expansion_mode, expansion_order,
               **kwargs):
        """
        Time evolution, exp^(-jt H).
        """
        raise NotImplementedError

    @abstractmethod
    def print_details(self):
        """ print details """
        raise NotImplementedError

    @abstractmethod
    def chop(self, threshold, copy=False):
        """ chop """
        raise NotImplementedError
