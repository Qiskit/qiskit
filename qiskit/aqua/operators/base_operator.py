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
import warnings

class BaseOperator(ABC):
    """Operators relevant for quantum applications."""

    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass

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

    @property
    def coloring(self):
        warnings.warn("coloring is removed, "
                      "Use the `TPBGroupedWeightedPauliOperator` class to group a paulis directly", DeprecationWarning)
        return None

    def _to_dia_matrix(self, mode=None):
        warnings.warn("_to_dia_matrix() is removed, use the `MatrixOperator` class instead", DeprecationWarning)

    def enable_summarize_circuits(self):
        warnings.warn("do not enable summary at the operator anymore, enable it at QuantumInstance", DeprecationWarning)

    def disable_summarize_circuits(self):
        warnings.warn("do not disable summary at the operator anymore, enable it at QuantumInstance", DeprecationWarning)

    @property
    def representations(self):
        warnings.warn("each operator is self-defined, no need to check represnetation anymore.", DeprecationWarning)
        return None

    def eval(self, operator_mode, input_circuit, backend, backend_config=None, compile_config=None,
             run_config=None, qjob_config=None, noise_config=None):
        warnings.warn("eval method is removed. please use `construct_evaluate_circuit` and submit circuit by yourself "
                      "then, use the result along with `evaluate_with_result` to get mean and std.", DeprecationWarning)
        return None, None

    def convert(self, input_format, output_format, force=False):
        warnings.warn("convert method is removed. please use to_XXX_operator in each operator class instead.",
                      DeprecationWarning)

    def two_qubit_reduced_operator(self, m, threshold=10 ** -13):
        warnings.warn("two_qubit_reduced_operator method is moved to the `TaperedWeightedPauliOperator` class.",
                      DeprecationWarning)
        return None

    def qubit_tapering(operator, cliffords, sq_list, tapering_values):
        warnings.warn("qubit_tapering method is moved to the `TaperedWeightedPauliOperator` class.",
                      DeprecationWarning)
        return None
