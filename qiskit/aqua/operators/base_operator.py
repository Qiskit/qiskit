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

from qiskit import QuantumCircuit


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
        return self._name

    @name.setter
    def name(self, new_value):
        self._name = new_value

    @property
    def basis(self):
        return self._basis

    @property
    def z2_symmetries(self):
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

    @abstractmethod
    def print_details(self):
        raise NotImplementedError

    @abstractmethod
    def _scaling_weight(self, scaling_factor):
        # TODO: will be removed after the deprecated method is removed.
        raise NotImplementedError

    @abstractmethod
    def chop(self, threshold, copy=False):
        raise NotImplementedError

    def print_operators(self, mode='paulis'):
        warnings.warn("print_operators() is deprecated and it will be removed after 0.6, "
                      "Use `print_details()` instead",
                      DeprecationWarning)
        return self.print_details()

    @property
    def coloring(self):
        warnings.warn("coloring is removed, "
                      "Use the `TPBGroupedWeightedPauliOperator` class to group a paulis directly",
                      DeprecationWarning)
        return None

    def _to_dia_matrix(self, mode=None):
        warnings.warn("_to_dia_matrix method is removed, use the `MatrixOperator` class to get diagonal matrix. And "
                      "the current deprecated method does NOT modify the original object, it returns the dia_matrix",
                      DeprecationWarning)
        from .op_converter import to_matrix_operator
        mat_op = to_matrix_operator(self)
        return mat_op.dia_matrix

    def enable_summarize_circuits(self):
        warnings.warn("enable_summarize_circuits method is removed. Enable the summary at QuantumInstance",
                      DeprecationWarning)

    def disable_summarize_circuits(self):
        warnings.warn("disable_summarize_circuits method is removed. Disable the summary at QuantumInstance",
                      DeprecationWarning)

    @property
    def representations(self):
        warnings.warn("representations method is removed. each operator is self-defined, ",
                      DeprecationWarning)
        return None

    def eval(self, operator_mode, input_circuit, backend, backend_config=None, compile_config=None,
             run_config=None, qjob_config=None, noise_config=None):
        warnings.warn("eval method is removed. please use `construct_evaluate_circuit` and submit circuit by yourself "
                      "then, use the result along with `evaluate_with_result` to get mean and std. "
                      "Furthermore, if you compute the expectation against a statevector (numpy array), you can "
                      "use evaluate_with_statevector directly.",
                      DeprecationWarning)
        return None, None

    def convert(self, input_format, output_format, force=False):
        warnings.warn("convert method is removed. please use the conversion functions in the "
                      "qiskit.aqua.operators.op_converter module. There are different `to_xxx_operator` functions"
                      " And the current deprecated method does NOT modify the original object, it returns.",
                      DeprecationWarning)
        from .op_converter import to_weighted_pauli_operator, to_matrix_operator, to_tpb_grouped_weighted_pauli_operator
        from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator
        if output_format == 'paulis':
            return to_weighted_pauli_operator(self)
        elif output_format == 'grouped_paulis':
            return to_tpb_grouped_weighted_pauli_operator(self, TPBGroupedWeightedPauliOperator.sorted_grouping)
        elif output_format == 'matrix':
            return to_matrix_operator(self)

    def two_qubit_reduced_operator(self, m, threshold=10 ** -13):
        warnings.warn("two_qubit_reduced_operator method is deprecated and it will be removed after 0.6. "
                      "Now it is moved to the `Z2Symmetries` class as a classmethod. """
                      "Z2Symmeteries.two_qubit_reduction(num_particles)",
                      DeprecationWarning)
        from .op_converter import to_weighted_pauli_operator
        from .weighted_pauli_operator import Z2Symmetries
        return Z2Symmetries.two_qubit_reduction(to_weighted_pauli_operator(self), m)

    @staticmethod
    def qubit_tapering(operator, cliffords, sq_list, tapering_values):
        warnings.warn("qubit_tapering method is deprecated and it will be removed after 0.6. "
                      "Now it is moved to the `Z2Symmetries` class.",
                      DeprecationWarning)
        from .op_converter import to_weighted_pauli_operator
        from .weighted_pauli_operator import Z2Symmetries
        sq_paulis = [x.paulis[1][1] for x in cliffords]
        symmetries = [x.paulis[0][1] for x in cliffords]
        tmp_op = to_weighted_pauli_operator(operator)
        z2_symmetries = Z2Symmetries(symmetries, sq_paulis, sq_list, tapering_values)
        return z2_symmetries.taper(tmp_op)

    def scaling_coeff(self, scaling_factor):
        warnings.warn("scaling_coeff method is deprecated and it will be removed after 0.6. "
                      "Use `* operator` with the scalar directly.",
                      DeprecationWarning)
        self._scaling_weight(scaling_factor)
        return self

    def zeros_coeff_elimination(self):
        warnings.warn("zeros_coeff_elimination method is deprecated and it will be removed after 0.6. "
                      "Use chop(0.0) to remove terms with 0 weight.",
                      DeprecationWarning)
        self.chop(0.0)
        return self

    @staticmethod
    def construct_evolution_circuit(slice_pauli_list, evo_time, num_time_slices, state_registers,
                                    ancillary_registers=None, ctl_idx=0, unitary_power=None, use_basis_gates=True,
                                    shallow_slicing=False):
        from .common import evolution_instruction
        warnings.warn("The `construct_evolution_circuit` method is deprecated, use the `evolution_instruction` in "
                      "the qiskit.aqua.operators.common module instead.",
                      DeprecationWarning)

        if state_registers is None:
            raise ValueError('Quantum state registers are required.')

        qc_slice = QuantumCircuit(state_registers)
        if ancillary_registers is not None:
            qc_slice.add_register(ancillary_registers)
        controlled = ancillary_registers is not None
        inst = evolution_instruction(slice_pauli_list, evo_time, num_time_slices, controlled, 2 ** ctl_idx,
                                     use_basis_gates, shallow_slicing)

        qc_slice.append(inst, [q for qreg in qc_slice.qregs for q in qreg])
        qc_slice = qc_slice.decompose()
        return qc_slice

    @staticmethod
    def row_echelon_F2(matrix_in):
        from .common import row_echelon_F2
        warnings.warn("The `row_echelon_F2` method is deprecated, use the row_echelon_F2 function in "
                      "the qiskit.aqua.operators.common module instead.",
                      DeprecationWarning)
        return row_echelon_F2(matrix_in)

    @staticmethod
    def kernel_F2(matrix_in):
        from .common import kernel_F2
        warnings.warn("The `kernel_F2` method is deprecated, use the kernel_F2 function in "
                      "the qiskit.aqua.operators.common module instead.",
                      DeprecationWarning)
        return kernel_F2(matrix_in)

    def find_Z2_symmetries(self):
        warnings.warn("The `find_Z2_symmetries` method is deprecated and it will be removed after 0.6, "
                      "Use the class method in the `Z2Symmetries` class instead",
                      DeprecationWarning)
        from .weighted_pauli_operator import Z2Symmetries
        from .op_converter import to_weighted_pauli_operator
        wp_op = to_weighted_pauli_operator(self)
        self._z2_symmetries = Z2Symmetries.find_Z2_symmetries(wp_op)

        return self._z2_symmetries.symmetries, self._z2_symmetries.sq_paulis, \
            self._z2_symmetries.cliffords, self._z2_symmetries.sq_list

    def to_grouped_paulis(self):
        warnings.warn("to_grouped_paulis method is deprecated and it will be removed after 0.6. And the current "
                      "deprecated method does NOT modify the original object, it returns the grouped weighted pauli "
                      "operator. Please check the qiskit.aqua.operators.op_convertor for converting to different "
                      "types of operators. For grouping paulis, you can create your own grouping func to create the "
                      "class you need.",
                      DeprecationWarning)
        from .op_converter import to_tpb_grouped_weighted_pauli_operator
        from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator
        return to_tpb_grouped_weighted_pauli_operator(self, grouping_func=TPBGroupedWeightedPauliOperator.sorted_grouping)

    def to_paulis(self):
        warnings.warn("to_paulis method is deprecated and it will be removed after 0.6. And the current deprecated "
                      "method does NOT modify the original object, it returns the weighted pauli operator."
                      "Please check the qiskit.aqua.operators.op_convertor for converting to different types of "
                      "operators",
                      DeprecationWarning)
        from .op_converter import to_weighted_pauli_operator
        return to_weighted_pauli_operator(self)

    def to_matrix(self):
        warnings.warn("to_matrix method is deprecated and it will be removed after 0.6. And the current deprecated "
                      "method does NOT modify the original object, it returns the matrix operator."
                      "Please check the qiskit.aqua.operators.op_convertor for converting to different types of "
                      "operators",
                      DeprecationWarning)
        from .op_converter import to_matrix_operator
        return to_matrix_operator(self)

    def to_weighted_pauli_operator(self):
        warnings.warn("to_weighted_apuli_operator method is temporary helper method and it will be removed after 0.6. "
                      "Please check the qiskit.aqua.operators.op_convertor for converting to different types of "
                      "operators",
                      DeprecationWarning)
        from .op_converter import to_weighted_pauli_operator
        return to_weighted_pauli_operator(self)

    def to_matrix_operator(self):
        warnings.warn("to_matrix_operator method is temporary helper method and it will be removed after 0.6. "
                      "Please check the qiskit.aqua.operators.op_convertor for converting to different types of "
                      "operators",
                      DeprecationWarning)
        from .op_converter import to_matrix_operator
        return to_matrix_operator(self)

    def to_tpb_grouped_weighted_pauli_operator(self):
        warnings.warn("to_tpb_grouped_weighted_pauli_operator method is temporary helper method and it will be "
                      "removed after 0.6. Please check the qiskit.aqua.operators.op_convertor for converting to "
                      "different types of operators",
                      DeprecationWarning)
        from .op_converter import to_tpb_grouped_weighted_pauli_operator
        from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator
        return to_tpb_grouped_weighted_pauli_operator(
            self, grouping_func=TPBGroupedWeightedPauliOperator.sorted_grouping)
