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

"""Methods for working with the CNOT-dihedral group.

Example:

  from dihedral import CNOTDihedral
  g = CNOTDihedral(3)  # create identity element on 3 qubits
  g.cnot(0,1)          # apply CNOT from qubit 0 to qubit 1
  g.flip(2)            # apply X on qubit 2
  g.phase(3, 1)        # apply T^3 on qubit 1
  print(g)             # pretty print g

  phase polynomial =
   0 + 3*x_0 + 3*x_1 + 2*x_0*x_1
  affine function =
   (x_0,x_0 + x_1,x_2 + 1)

 This means that |x_0 x_1 x_2> transforms to omega^{p(x)}|f(x)>,
 where omega = exp(i*pi/4) from which we can read that
 T^3 on qubit 1 AFTER CNOT_{0,1} is the same as
 T^3 on qubit 0, T^3 on qubit 1, and CS_{0,1} BEFORE CNOT_{0,1}.
"""

import itertools
from itertools import combinations
import copy
from functools import reduce
from operator import mul
import numpy as np
from numpy.random import RandomState

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library import U1Gate


class SpecialPolynomial():
    """Multivariate polynomial with special form.

    Maximum degree 3, n Z_2 variables, coefficients in Z_8.
    """

    def __init__(self, n_vars):
        """Construct the zero polynomial on n_vars variables."""
        #   1 constant term
        #   n linear terms x_1, ..., x_n
        #   {n choose 2} quadratic terms x_1x_2, x_1x_3, ..., x_{n-1}x_n
        #   {n choose 3} cubic terms x_1x_2x_3, ..., x_{n-2}x_{n-1}x_n
        # and coefficients in Z_8
        if n_vars < 1:
            raise QiskitError("n_vars for SpecialPolynomial is too small.")
        self.n_vars = n_vars
        self.nc2 = int(n_vars * (n_vars-1) / 2)
        self.nc3 = int(n_vars * (n_vars-1) * (n_vars-2) / 6)
        self.weight_0 = 0
        self.weight_1 = np.zeros(n_vars, dtype=np.int8)
        self.weight_2 = np.zeros(self.nc2, dtype=np.int8)
        self.weight_3 = np.zeros(self.nc3, dtype=np.int8)

    def mul_monomial(self, indices):
        """Multiply by a monomial given by indices.

        Returns the product.
        """
        length = len(indices)
        if length >= 4:
            raise QiskitError("There is no term with on more than 3 indices.")
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() and (indices_arr > self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        if length > 1 and (np.diff(indices_arr) <= 0).any():
            raise QiskitError("Indices are non-increasing!")
        result = SpecialPolynomial(self.n_vars)
        if length == 0:
            result = copy.deepcopy(self)
        else:
            terms0 = [[]]
            terms1 = list(combinations(range(self.n_vars), r=1))
            terms2 = list(combinations(range(self.n_vars), r=2))
            terms3 = list(combinations(range(self.n_vars), r=3))
            for term in terms0 + terms1 + terms2 + terms3:
                value = self.get_term(term)
                new_term = list(set(term).union(set(indices)))
                result.set_term(new_term, (result.get_term(new_term) + value) % 8)
        return result

    def __mul__(self, other):
        """Multiply two polynomials."""
        if not isinstance(other, SpecialPolynomial):
            other = int(other)
        result = SpecialPolynomial(self.n_vars)
        if isinstance(other, int):
            result.weight_0 = (self.weight_0 * other) % 8
            result.weight_1 = (self.weight_1 * other) % 8
            result.weight_2 = (self.weight_2 * other) % 8
            result.weight_3 = (self.weight_3 * other) % 8
        else:
            if self.n_vars != other.n_vars:
                raise QiskitError("Multiplication on different n_vars.")
            terms0 = [[]]
            terms1 = list(combinations(range(self.n_vars), r=1))
            terms2 = list(combinations(range(self.n_vars), r=2))
            terms3 = list(combinations(range(self.n_vars), r=3))
            for term in terms0 + terms1 + terms2 + terms3:
                value = other.get_term(term)
                if value != 0:
                    temp = copy.deepcopy(self)
                    temp = temp.mul_monomial(term)
                    temp = temp * value
                    result = result + temp
        return result

    def __rmul__(self, other):
        """Right multiplication.

        This operation is commutative.
        """
        return self.__mul__(other)

    def __add__(self, other):
        """Add two polynomials."""
        if not isinstance(other, SpecialPolynomial):
            raise QiskitError("Element to add is not a SpecialPolynomial.")
        if self.n_vars != other.n_vars:
            raise QiskitError("Addition on different n_vars.")
        result = SpecialPolynomial(self.n_vars)
        result.weight_0 = (self.weight_0 + other.weight_0) % 8
        result.weight_1 = (self.weight_1 + other.weight_1) % 8
        result.weight_2 = (self.weight_2 + other.weight_2) % 8
        result.weight_3 = (self.weight_3 + other.weight_3) % 8
        return result

    def evaluate(self, xval):
        """Evaluate the multinomial at xval.

        if xval is a length n z2 vector, return element of Z8.
        if xval is a length n vector of multinomials, return
        a multinomial. The multinomials must all be on n vars.
        """
        if len(xval) != self.n_vars:
            raise QiskitError("Evaluate on wrong number of variables.")
        check_int = list(map(lambda x: isinstance(x, int), xval))
        check_poly = list(map(lambda x: isinstance(x, SpecialPolynomial), xval))
        if False in check_int and False in check_poly:
            raise QiskitError("Evaluate on a wrong type.")
        is_int = (False not in check_int)
        if not is_int:
            if False in [i.n_vars == self.n_vars for i in xval]:
                raise QiskitError("Evaluate on incompatible polynomials.")
        else:
            xval = xval % 2
        # Examine each term of this polynomial
        terms0 = [[]]
        terms1 = list(combinations(range(self.n_vars), r=1))
        terms2 = list(combinations(range(self.n_vars), r=2))
        terms3 = list(combinations(range(self.n_vars), r=3))
        # Set the initial result and start for each term
        if is_int:
            result = 0
            start = 1
        else:
            result = SpecialPolynomial(self.n_vars)
            start = SpecialPolynomial(self.n_vars)
            start.weight_0 = 1
        # Compute the new terms and accumulate
        for term in terms0 + terms1 + terms2 + terms3:
            value = self.get_term(term)
            if value != 0:
                newterm = reduce(mul, [xval[j] for j in term], start)
                result = result + value * newterm
        if isinstance(result, int):
            result = result % 8
        return result

    def set_pj(self, indices):
        """Set to special form polynomial on subset of variables.

        p_J(x) := sum_{a subseteq J,|a| neq 0} (-2)^{|a|-1}x^a
        """
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        indices = sorted(indices)
        subsets_2 = itertools.combinations(indices, 2)
        subsets_3 = itertools.combinations(indices, 3)
        self.weight_0 = 0
        self.weight_1 = np.zeros(self.n_vars)
        self.weight_2 = np.zeros(self.nc2)
        self.weight_3 = np.zeros(self.nc3)
        for j in indices:
            self.set_term([j], 1)
        for j in subsets_2:
            self.set_term(list(j), 6)
        for j in subsets_3:
            self.set_term(list(j), 4)

    def get_term(self, indices):
        """Get the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        """
        length = len(indices)
        if length >= 4:
            return 0
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        if length > 1 and (np.diff(indices_arr) <= 0).any():
            raise QiskitError("Indices are non-increasing.")

        if length == 0:
            return self.weight_0
        if length == 1:
            return self.weight_1[indices[0]]
        if length == 2:
            # sum(self.n_vars-j, {j, 1, indices[0]})
            offset_1 = int(indices[0] * self.n_vars -
                           ((indices[0] + 1) * indices[0])/2)
            offset_2 = int(indices[1] - indices[0] - 1)
            return self.weight_2[offset_1 + offset_2]

        # handle length = 3
        tmp_1 = self.n_vars - indices[0]
        offset_1 = int((tmp_1 - 3) * (tmp_1 - 2) * (tmp_1 - 1) / 6)
        tmp_2 = self.n_vars - indices[1]
        offset_2 = int((tmp_2 - 2) * (tmp_2 - 1) / 2)
        offset_3 = self.n_vars - indices[2]
        offset = int(self.n_vars * (self.n_vars - 1) * (self.n_vars - 2) / 6 -
                     offset_1 - offset_2 - offset_3)

        return self.weight_3[offset]

    def set_term(self, indices, value):
        """Set the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        The value is reduced modulo 8.
        """
        length = len(indices)
        if length >= 4:
            return
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        if length > 1 and (np.diff(indices_arr) <= 0).any():
            raise QiskitError("Indices are non-increasing.")

        value = value % 8
        if length == 0:
            self.weight_0 = value
        elif length == 1:
            self.weight_1[indices[0]] = value
        elif length == 2:
            # sum(self.n_vars-j, {j, 1, indices[0]})
            offset_1 = int(indices[0] * self.n_vars -
                           ((indices[0] + 1) * indices[0])/2)
            offset_2 = int(indices[1] - indices[0] - 1)
            self.weight_2[offset_1 + offset_2] = value
        else:  # length = 3
            tmp_1 = self.n_vars - indices[0]
            offset_1 = int((tmp_1 - 3) * (tmp_1 - 2) * (tmp_1 - 1) / 6)
            tmp_2 = self.n_vars - indices[1]
            offset_2 = int((tmp_2 - 2) * (tmp_2 - 1) / 2)
            offset_3 = self.n_vars - indices[2]
            offset = int(self.n_vars * (self.n_vars - 1) * (self.n_vars - 2) / 6 -
                         offset_1 - offset_2 - offset_3)
            self.weight_3[offset] = value

    @property
    def key(self):
        """Return a string representation."""
        tup = (self.weight_0, tuple(self.weight_1),
               tuple(self.weight_2), tuple(self.weight_3))
        return str(tup)

    def __eq__(self, x):
        """Test equality."""
        return isinstance(x, SpecialPolynomial) and self.key == x.key

    def __str__(self):
        """Return formatted string representation."""
        out = str(self.weight_0)
        for i in range(self.n_vars):
            value = self.get_term([i])
            if value != 0:
                out += " + "
                if value != 1:
                    out += (str(value) + "*")
                out += ("x_" + str(i))
        for i in range(self.n_vars-1):
            for j in range(i+1, self.n_vars):
                value = self.get_term([i, j])
                if value != 0:
                    out += " + "
                    if value != 1:
                        out += (str(value) + "*")
                    out += ("x_" + str(i) + "*x_" + str(j))
        for i in range(self.n_vars-2):
            for j in range(i+1, self.n_vars-1):
                for k in range(j+1, self.n_vars):
                    value = self.get_term([i, j, k])
                    if value != 0:
                        out += " + "
                        if value != 1:
                            out += (str(value) + "*")
                        out += ("x_" + str(i) + "*x_" + str(j) +
                                "*x_" + str(k))
        return out


class CNOTDihedral(BaseOperator):
    """CNOT-dihedral Object Class.
    The CNOT-dihedral group on num_qubits qubits is generated by the gates
    CNOT, T and X.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    def __init__(self, data, validate=True):
        """Initialize a CNOTDihedral operator object."""

        # Initialize from another CNOTDihedral by sharing the underlying
        # poly, linear and shift
        if isinstance(data, CNOTDihedral):
            self.linear = data.linear
            self.shift = data.shift
            self.poly = data.poly

        # Initialize from ScalarOp as N-qubit identity discarding any global phase
        elif isinstance(data, ScalarOp):
            if not data.is_unitary() or set(data._input_dims) != {2} or \
                    data.num_qubits is None:
                raise QiskitError("Can only initialize from N-qubit identity ScalarOp.")
            self._num_qubits = data.num_qubits
            # phase polynomial
            self.poly = SpecialPolynomial(self._num_qubits)
            # n x n invertible matrix over Z_2
            self.linear = np.eye(self._num_qubits, dtype=np.int8)
            # binary shift, n coefficients in Z_2
            self.shift = np.zeros(self._num_qubits, dtype=np.int8)

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._num_qubits = data.num_qubits
            elem = self.from_circuit(data)
            self.poly = elem.poly
            self.linear = elem.linear
            self.shift = elem.shift

        # Construct the identity element on num_qubits qubits.
        elif isinstance(data, int):
            self._num_qubits = data
            # phase polynomial
            self.poly = SpecialPolynomial(self._num_qubits)
            # n x n invertible matrix over Z_2
            self.linear = np.eye(self._num_qubits, dtype=np.int8)
            # binary shift, n coefficients in Z_2
            self.shift = np.zeros(self._num_qubits, dtype=np.int8)

        elif isinstance(data, Pauli):
            self._num_qubits = data.num_qubits
            elem = self.from_circuit(data.to_instruction())
            self.poly = elem.poly
            self.linear = elem.linear
            self.shift = elem.shift

        # Initialize BaseOperator
        dims = self._num_qubits * (2,)
        super().__init__(dims, dims)

        # Validate the CNOTDihedral element
        if validate and not self.is_cnotdihedral():
            raise QiskitError('Invalid CNOTDihedsral element.')

    def _z2matmul(self, left, right):
        """Compute product of two n x n z2 matrices."""
        prod = np.mod(np.dot(left, right), 2)
        return prod

    def _z2matvecmul(self, mat, vec):
        """Compute mat*vec of n x n z2 matrix and vector."""
        prod = np.mod(np.dot(mat, vec), 2)
        return prod

    def __mul__(self, other):
        """Left multiplication self * other."""
        if self.num_qubits != other.num_qubits:
            raise QiskitError("Multiplication on different number of qubits.")
        result = CNOTDihedral(self.num_qubits)
        result.shift = [(x[0] + x[1]) % 2
                        for x in zip(self._z2matvecmul(self.linear, other.shift), self.shift)]
        result.linear = self._z2matmul(self.linear, other.linear)
        # Compute x' = B1*x + c1 using the p_j identity
        new_vars = []
        for i in range(self.num_qubits):
            support = np.arange(self.num_qubits)[np.nonzero(other.linear[i])]
            poly = SpecialPolynomial(self.num_qubits)
            poly.set_pj(support)
            if other.shift[i] == 1:
                poly = -1 * poly
                poly.weight_0 = (poly.weight_0 + 1) % 8
            new_vars.append(poly)
        # p' = p1 + p2(x')
        result.poly = other.poly + self.poly.evaluate(new_vars)
        return result

    def __rmul__(self, other):
        """Right multiplication other * self."""
        if self.num_qubits != other.num_qubits:
            raise QiskitError("Multiplication on different number of qubits.")
        result = CNOTDihedral(self.num_qubits)
        result.shift = [(x[0] + x[1]) % 2
                        for x in zip(self._z2matvecmul(other.linear,
                                                       self.shift),
                                     other.shift)]
        result.linear = self._z2matmul(other.linear, self.linear)
        # Compute x' = B1*x + c1 using the p_j identity
        new_vars = []
        for i in range(self.num_qubits):
            support = np.arange(self.num_qubits)[np.nonzero(self.linear[i])]
            poly = SpecialPolynomial(self.num_qubits)
            poly.set_pj(support)
            if other.shift[i] == 1:
                poly = -1 * poly
                poly.weight_0 = (poly.weight_0 + 1) % 8
            new_vars.append(poly)
        # p' = p1 + p2(x')
        result.poly = self.poly + other.poly.evaluate(new_vars)
        return result

    @property
    def key(self):
        """Return a string representation of a CNOT-dihedral object."""
        tup = (self.poly.key, tuple(map(tuple, self.linear)),
               tuple(self.shift))
        return str(tup)

    def __eq__(self, x):
        """Test equality."""
        return isinstance(x, CNOTDihedral) and self.key == x.key

    def cnot(self, i, j):
        """Apply a CNOT gate to this element.
        Left multiply the element by CNOT_{i,j}.
        """

        if not 0 <= i < self.num_qubits or not 0 <= j < self.num_qubits:
            raise QiskitError("cnot qubits are out of bounds.")
        self.linear[j] = (self.linear[i] + self.linear[j]) % 2
        self.shift[j] = (self.shift[i] + self.shift[j]) % 2

    def phase(self, k, i):
        """Apply an k-th power of T to this element.
        Left multiply the element by T_i^k.
        """
        if not 0 <= i < self.num_qubits:
            raise QiskitError("phase qubit out of bounds.")
        # If the kth bit is flipped, conjugate this gate
        if self.shift[i] == 1:
            k = (7*k) % 8
        # Take all subsets \alpha of the support of row i
        # of weight up to 3 and add k*(-2)**(|\alpha| - 1) mod 8
        # to the corresponding term.
        support = np.arange(self.num_qubits)[np.nonzero(self.linear[i])]
        subsets_2 = itertools.combinations(support, 2)
        subsets_3 = itertools.combinations(support, 3)
        for j in support:
            value = self.poly.get_term([j])
            self.poly.set_term([j], (value + k) % 8)
        for j in subsets_2:
            value = self.poly.get_term(list(j))
            self.poly.set_term(list(j), (value + -2 * k) % 8)
        for j in subsets_3:
            value = self.poly.get_term(list(j))
            self.poly.set_term(list(j), (value + 4 * k) % 8)

    def flip(self, i):
        """Apply X to this element.
        Left multiply the element by X_i.
        """
        if not 0 <= i < self.num_qubits:
            raise QiskitError("flip qubit out of bounds.")
        self.shift[i] = (self.shift[i] + 1) % 2

    def __str__(self):
        """Return formatted string representation."""
        out = "phase polynomial = \n"
        out += str(self.poly)
        out += "\naffine function = \n"
        out += " ("
        for row in range(self.num_qubits):
            wrote = False
            for col in range(self.num_qubits):
                if self.linear[row][col] != 0:
                    if wrote:
                        out += (" + x_" + str(col))
                    else:
                        out += ("x_" + str(col))
                        wrote = True
            if self.shift[row] != 0:
                out += " + 1"
            if row != self.num_qubits - 1:
                out += ","
        out += ")\n"
        return out

    def _add(self, other, qargs=None):
        """Not implemented."""
        raise NotImplementedError(
            "{} does not support addition".format(type(self)))

    def _multiply(self, other):
        """Not implemented."""
        raise NotImplementedError(
            "{} does not support scalar multiplication".format(type(self)))

    def to_circuit(self):
        """Return a QuantumCircuit implementing the CNOT-Dihedral element.

        Return:
            QuantumCircuit: a circuit implementation of the CNOTDihedral object.
        Remark:
            Decompose 1 and 2-qubit CNOTDihedral elements.

        References:
            1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
               `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
            2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
               *Scalable randomised benchmarking of non-Clifford gates*,
               npj Quantum Inf 2, 16012 (2016).
        """
        return decompose_cnotdihedral(self)

    def to_instruction(self):
        """Return a Gate instruction implementing the CNOTDihedral object."""
        return self.to_circuit().to_gate()

    def from_circuit(self, circuit):
        """Initialize from a QuantumCircuit or Instruction.

        Args:
            circuit (QuantumCircuit or ~qiskit.circuit.Instruction):
                instruction to initialize.
        Returns:
            CNOTDihedral: the CNOTDihedral object for the circuit.
        Raises:
            QiskitError: if the input instruction is not CNOTDihedral or contains
                         classical register instruction.
        """
        if not isinstance(circuit, (QuantumCircuit, Instruction)):
            raise QiskitError("Input must be a QuantumCircuit or Instruction")

        # Convert circuit to an instruction
        if isinstance(circuit, QuantumCircuit):
            circuit = circuit.to_instruction()

        # Initialize an identity CNOTDihedral object
        elem = CNOTDihedral(self.num_qubits)
        append_circuit(elem, circuit)
        return elem

    def to_matrix(self):
        """Convert operator to Numpy matrix."""
        return self.to_operator().data

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_instruction())

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (CNOTDihedral): an operator object.
            qargs (None): using specific qargs is not implemented for this operator.
            front (bool): if True compose using right operator multiplication,
                          instead of left multiplication [default: False].
        Returns:
            CNOTDihedral: The operator self @ other.
        Raises:
            QiskitError: if operators have incompatible dimensions for
                         composition.
            NotImplementedError: if qargs is not None.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is not None:
            raise NotImplementedError("compose method does not support qargs.")
        if self.num_qubits != other.num_qubits:
            raise QiskitError("Incompatible dimension for composition")
        if front:
            other = self * other
        else:
            other = other * self
        other.poly.weight_0 = 0  # set global phase
        return other

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (CNOTDihedral): an operator object.
            qargs (None): using specific qargs is not implemented for this operator.
        Returns:
            CNOTDihedral: The operator self * other.
        Raises:
            QiskitError: if operators have incompatible dimensions for composition.
            NotImplementedError: if qargs is not None.
        """
        if qargs is not None:
            raise NotImplementedError("dot method does not support qargs.")
        if self.num_qubits != other.num_qubits:
            raise QiskitError("Incompatible dimension for composition")
        other = self * other
        other.poly.weight_0 = 0  # set global phase
        return other

    def _tensor_product(self, other, reverse=False):
        """Returns the tensor product operator.

         Args:
             other (CNOTDihedral): another operator subclass object.
             reverse (bool): If False return self tensor other,
                            if True return other tensor self [Default: False].
         Returns:
             CNOTDihedral: the tensor product operator: self tensor other.
         Raises:
             QiskitError: if other cannot be converted into an CNOTDihderal object.
        """

        if not isinstance(other, CNOTDihedral):
            raise QiskitError("Tensored element is not a CNOTDihderal object.")

        if reverse:
            elem0 = self
            elem1 = other
        else:
            elem0 = other
            elem1 = self

        result = CNOTDihedral(elem0.num_qubits + elem1.num_qubits)
        linear = np.block([[elem0.linear,
                            np.zeros((elem0.num_qubits, elem1.num_qubits), dtype=np.int8)],
                           [np.zeros((elem1.num_qubits, elem0.num_qubits), dtype=np.int8),
                            elem1.linear]])
        result.linear = linear
        shift = np.block([elem0.shift, elem1.shift])
        result.shift = shift

        for i in range(elem0.num_qubits):
            value = elem0.poly.get_term([i])
            result.poly.set_term([i], value)
            for j in range(i):
                value = elem0.poly.get_term([j, i])
                result.poly.set_term([j, i], value)
                for k in range(j):
                    value = elem0.poly.get_term([k, j, i])
                    result.poly.set_term([k, j, i], value)

        for i in range(elem1.num_qubits):
            value = elem1.poly.get_term([i])
            result.poly.set_term([i + elem0.num_qubits], value)
            for j in range(i):
                value = elem1.poly.get_term([j, i])
                result.poly.set_term([j + elem0.num_qubits, i + elem0.num_qubits], value)
                for k in range(j):
                    value = elem1.poly.get_term([k, j, i])
                    result.poly.set_term([k + elem0.num_qubits, j + elem0.num_qubits,
                                          i + elem0.num_qubits], value)

        return result

    def tensor(self, other):
        """Return the tensor product operator: self tensor other.

         Args:
             other (CNOTDihedral): an operator subclass object.
         Returns:
             CNOTDihedral: the tensor product operator: self tensor other.
         """

        return self._tensor_product(other, reverse=True)

    def expand(self, other):
        """Return the tensor product operator: other tensor self.

         Args:
             other (CNOTDihedral): an operator subclass object.
         Returns:
             CNOTDihedral: the tensor product operator: other tensor other.
         """

        return self._tensor_product(other, reverse=False)

    def adjoint(self):
        """Return the conjugate transpose of the CNOTDihedral element"""

        circ = self.to_instruction()
        result = self.from_circuit(circ.inverse())
        return result

    def conjugate(self):
        """Return the conjugate of the CNOTDihedral element."""
        circ = self.to_instruction()
        new_circ = QuantumCircuit(self.num_qubits)
        qargs = list(range(self.num_qubits))
        for instr, qregs, _ in circ.definition:
            new_qubits = [qargs[tup.index] for tup in qregs]
            if instr.name == 'u1':
                params = 2 * np.pi - instr.params[0]
                instr.params[0] = params
                new_circ.append(instr, new_qubits)
            else:
                new_circ.append(instr, new_qubits)
        result = self.from_circuit(new_circ)
        return result

    def transpose(self):
        """Return the transpose of the CNOT-Dihedral element."""

        circ = self.to_instruction()
        result = self.from_circuit(circ.reverse_ops())
        return result

    def is_cnotdihedral(self):
        """Return True if input is a CNOTDihedral element."""

        if self.poly.weight_0 != 0 or \
                len(self.poly.weight_1) != self.num_qubits or \
                len(self.poly.weight_2) != int(self.num_qubits * (self.num_qubits - 1) / 2) \
                or len(self.poly.weight_3) != int(self.num_qubits * (self.num_qubits - 1)
                                                  * (self.num_qubits - 2) / 6):
            return False
        if (self.linear).shape != (self.num_qubits, self.num_qubits) or \
                len(self.shift) != self.num_qubits or \
                not np.allclose((np.linalg.det(self.linear) % 2), 1):
            return False
        if not (set(self.poly.weight_1.flatten())).issubset({0, 1, 2, 3, 4, 5, 6, 7}) or \
                not (set(self.poly.weight_2.flatten())).issubset({0, 2, 4, 6}) or \
                not (set(self.poly.weight_3.flatten())).issubset({0, 4}):
            return False
        if not (set(self.shift.flatten())).issubset({0, 1}) or \
                not (set(self.linear.flatten())).issubset({0, 1}):
            return False
        return True


def make_dict_0(num_qubits):
    """Make the zero-CNOT dictionary.

    This returns the dictionary of CNOT-dihedral elements on
    num_qubits using no CNOT gates. There are 16^n elements.
    The key is a unique string and the value is a pair:
    a CNOTDihedral object and a list of gates as a string.
    """
    obj = {}
    for i in range(16**num_qubits):
        elem = CNOTDihedral(num_qubits)
        circ = []
        num = i
        for j in range(num_qubits):
            xpower = int(num % 2)
            tpower = int(((num - num % 2) / 2) % 8)
            if tpower > 0:
                elem.phase(tpower, j)
                circ.append(("u1", tpower, j))
            if xpower == 1:
                elem.flip(j)
                circ.append(("x", j))
            num = int((num - num % 16) / 16)
        obj[elem.key] = (elem, circ)

    return obj


def make_dict_next(num_qubits, dicts_prior):
    """Make the m+1 CNOT dictionary given the prior dictionaries.

    This returns the dictionary of CNOT-dihedral elements on
    num_qubits using m+1 CNOT gates given the list of dictionaries
    of circuits using 0, 1, ..., m CNOT gates.
    There are no more than 4*(n^2 - n)*|G(m)| elements.
    The key is a unique string and the value is a pair:
    a CNOTDihedral object and a list of gates as a string.
    """
    obj = {}
    for elem, circ in dicts_prior[-1].values():
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    for tpower in range(4):
                        new_elem = copy.deepcopy(elem)
                        new_circ = copy.deepcopy(circ)
                        new_elem.cnot(i, j)
                        new_circ.append(("cx", i, j))
                        if tpower > 0:
                            new_elem.phase(tpower, j)
                            new_circ.append(("u1", tpower, j))
                        if True not in [(new_elem.key in d)
                                        for d in dicts_prior] \
                                and new_elem.key not in obj:
                            obj[new_elem.key] = (new_elem, new_circ)

    return obj


def append_circuit(elem, circuit, qargs=None):
    """Update a CNOTDihedral element inplace by applying a CNOTDihedral circuit.

    Args:
        elem (CNOTDihedral): the CNOTDihedral element to update.
        circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
        qargs (list or None): The qubits to apply gates to.
    Returns:
        CNOTDihedral: the updated CNOTDihedral.
    Raises:
        QiskitError: if input gates cannot be decomposed into CNOTDihedral gates.
    """

    if qargs is None:
        qargs = list(range(elem.num_qubits))

    if isinstance(circuit, QuantumCircuit):
        gate = circuit.to_instruction()
    else:
        gate = circuit

    # Handle cx since it is a basic gate, and cannot be decomposed,
    # so gate.definition = None
    if gate.name == 'cx':
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate cx.")
        elem.cnot(qargs[0], qargs[1])
        return elem

    if gate.definition is None:
        raise QiskitError('Cannot apply Instruction: {}'.format(gate.name))
    if not isinstance(gate.definition, QuantumCircuit):
        raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(
            gate.name, type(gate.definition)))

    for instr, qregs, _ in gate.definition:
        # Get the integer position of the flat register
        new_qubits = [qargs[tup.index] for tup in qregs]

        if (instr.name == 'x' or gate.name == 'x'):
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate x.")
            elem.flip(new_qubits[0])

        elif (instr.name == 'z' or gate.name == 'z'):
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate z.")
            elem.phase(4, new_qubits[0])

        elif (instr.name == 'y' or gate.name == 'y'):
            if len(new_qubits) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate y.")
            elem.flip(new_qubits[0])
            elem.phase(4, new_qubits[0])

        elif (instr.name == 'u1' or gate.name == 'u1'):
            if (len(new_qubits) != 1 or len(instr.params) != 1):
                raise QiskitError("Invalid qubits or params for 1-qubit gate u1.")
            elem.phase(int(4 * instr.params[0] / np.pi), new_qubits[0])

        elif (instr.name == 'cx' or gate.name == 'cx'):
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate cx.")
            elem.cnot(new_qubits[0], new_qubits[1])

        elif (instr.name == 'cz' or gate.name == 'cz'):
            if len(new_qubits) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate cz.")
            elem.phase(7, new_qubits[1])
            elem.phase(7, new_qubits[0])
            elem.cnot(new_qubits[1], new_qubits[0])
            elem.phase(2, new_qubits[0])
            elem.cnot(new_qubits[1], new_qubits[0])
            elem.phase(7, new_qubits[1])
            elem.phase(7, new_qubits[0])

        elif (instr.name == 'id' or gate.name == 'id'):
            pass

        else:
            raise QiskitError('Not a CNOT-Dihedral gate: {}'.format(gate.name))

    return elem


def decompose_cnotdihedral(elem):
    """Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    num_qubits = elem.num_qubits

    if num_qubits < 3:
        return decompose_cnotdihedral_2_qubits(elem)

    return decompose_cnotdihedral_general(elem)


def decompose_cnotdihedral_2_qubits(elem):
    """Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.
    Remark:
        Decompose 1 and 2-qubit CNOTDihedral elements.
    Raises:
        QiskitError: if the element in not 1 or 2-qubit CNOTDihedral.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    circuit = QuantumCircuit(elem.num_qubits)

    if elem.num_qubits > 2:
        raise QiskitError("Cannot decompose a CNOT-Dihedral element with more than 2 qubits. "
                          "use decompose_cnotdihedral_general function instead.")

    if elem.num_qubits == 1:
        if elem.poly.weight_0 != 0 or elem.linear != [[1]]:
            raise QiskitError("1-qubit element in not CNOT-Dihedral .")
        tpow0 = elem.poly.weight_1[0]
        xpow0 = elem.shift[0]
        if tpow0 > 0:
            circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if (tpow0 == 0 and xpow0 == 0):
            circuit.id(0)
        return circuit

    # case elem.num_qubits == 2:
    if elem.poly.weight_0 != 0:
        raise QiskitError("2-qubit element in not CNOT-Dihedral .")
    weight_1 = elem.poly.weight_1
    weight_2 = elem.poly.weight_2
    linear = elem.linear
    shift = elem.shift

    # CS subgroup
    if (linear == [[1, 0], [0, 1]]).all():
        [xpow0, xpow1] = shift

        # Dihedral class
        if weight_2 == [0]:
            [tpow0, tpow1] = weight_1
            if tpow0 > 0:
                circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            if (tpow0 == 0 and tpow1 == 0 and xpow0 == 0 and xpow1 == 0):
                circuit.id(0)
                circuit.id(1)

        # CS-like class
        if ((weight_2 == [2] and xpow0 == xpow1) or
                (weight_2 == [6] and xpow0 != xpow1)):
            tpow0 = (weight_1[0] - 2 * xpow1 - 4 * xpow0 * xpow1) % 8
            tpow1 = (weight_1[1] - 2 * xpow0 - 4 * xpow0 * xpow1) % 8
            if tpow0 > 0:
                circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CS gate is implemented using 2 CX gates
            circuit.append(U1Gate(np.pi / 4), [0])
            circuit.append(U1Gate(np.pi / 4), [1])
            circuit.cx(0, 1)
            circuit.append(U1Gate(7 * np.pi / 4), [1])
            circuit.cx(0, 1)

        # CSdg-like class
        if ((weight_2 == [6] and xpow0 == xpow1) or
                (weight_2 == [2] and xpow0 != xpow1)):
            tpow0 = (weight_1[0] - 6 * xpow1 - 4 * xpow0 * xpow1) % 8
            tpow1 = (weight_1[1] - 6 * xpow0 - 4 * xpow0 * xpow1) % 8
            if tpow0 > 0:
                circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CSdg gate is implemented using 2 CX gates
            circuit.append(U1Gate(7 * np.pi / 4), [0])
            circuit.append(U1Gate(7 * np.pi / 4), [1])
            circuit.cx(0, 1)
            circuit.append(U1Gate(np.pi / 4), [1])
            circuit.cx(0, 1)

        # CZ-like class
        if weight_2 == [4]:
            tpow0 = (weight_1[0] - 4 * xpow1) % 8
            tpow1 = (weight_1[1] - 4 * xpow0) % 8
            if tpow0 > 0:
                circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CZ gate is implemented using 2 CX gates
            circuit.cz(1, 0)

    # CX01-like class
    if (linear == [[1, 0], [1, 1]]).all():
        xpow0 = shift[0]
        xpow1 = (shift[1] + xpow0) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        if m > 0:
            circuit.append(U1Gate(m * np.pi / 4), [1])

    # CX10-like class
    if (linear == [[1, 1], [0, 1]]).all():
        xpow1 = shift[1]
        xpow0 = (shift[0] + xpow1) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.append(U1Gate(m * np.pi / 4), [0])

    # CX01*CX10-like class
    if (linear == [[0, 1], [1, 1]]).all():
        xpow1 = shift[0]
        xpow0 = (shift[1] + xpow1) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.append(U1Gate(m * np.pi / 4), [1])

    # CX10*CX01-like class
    if (linear == [[1, 1], [1, 0]]).all():
        xpow0 = shift[1]
        xpow1 = (shift[0] + xpow0) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        if m > 0:
            circuit.append(U1Gate(m * np.pi / 4), [0])

    # CX01*CX10*CX01-like class
    if (linear == [[0, 1], [1, 0]]).all():
        xpow0 = shift[1]
        xpow1 = shift[0]
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.append(U1Gate(tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.append(U1Gate(tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.append(U1Gate(m * np.pi / 4), [1])
        circuit.cx(0, 1)

    return circuit


def decompose_cnotdihedral_general(elem):
    """Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.
    Remark:
        Decompose general CNOTDihedral elements.
        The number of CNOT gates is not necessarily optimal.
    Raises:
        QiskitError: if the element could not be decomposed into a circuit.

    References:
        1. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    num_qubits = elem.num_qubits
    circuit = QuantumCircuit(num_qubits)

    # Make a copy of the CNOTDihedral element as we are going to
    # reduce it to an identity
    elem_cpy = elem.copy()

    if not np.allclose((np.linalg.det(elem_cpy.linear) % 2), 1):
        raise QiskitError("Linear part is not invertible.")

    # Do x gate for each qubit i where shift[i]=1
    for i in range(num_qubits):
        if elem.shift[i]:
            circuit.x(i)
            elem_cpy.flip(i)

    # Do Gauss elimination on the linear part by adding cx gates
    for i in range(num_qubits):
        # set i-th element to be 1
        if not elem_cpy.linear[i][i]:
            for j in range(i+1, num_qubits):
                if elem_cpy.linear[j][i]:  # swap qubits i and j
                    circuit.cx(j, i)
                    circuit.cx(i, j)
                    circuit.cx(j, i)
                    elem_cpy.cnot(j, i)
                    elem_cpy.cnot(i, j)
                    elem_cpy.cnot(j, i)
                    break
        # make all the other elements in column i zero
        for j in range(num_qubits):
            if j != i:
                if elem_cpy.linear[j][i]:
                    circuit.cx(i, j)
                    elem_cpy.cnot(i, j)

    if not (elem_cpy.shift == np.zeros(num_qubits)).all() or \
            not (elem_cpy.linear == np.eye(num_qubits)).all():
        raise QiskitError("Cannot do Gauss elimination on linear part.")

    new_elem = CNOTDihedral(num_qubits)
    new_circuit = QuantumCircuit(num_qubits)

    # Do cx and u1 gates to construct all monomials of weight 3
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                if elem_cpy.poly.get_term([i, j, k]) != 0:
                    new_elem.cnot(i, k)
                    new_elem.cnot(j, k)
                    new_elem.phase(1, k)
                    new_elem.cnot(i, k)
                    new_elem.cnot(j, k)
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)
                    new_circuit.append(U1Gate(np.pi / 4), [k])
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)

    # Do cx and u1 gates to construct all monomials of weight 2
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            tpow1 = elem_cpy.poly.get_term([i, j])
            tpow2 = new_elem.poly.get_term([i, j])
            tpow = ((tpow2 - tpow1) / 2) % 4
            if tpow != 0:
                new_elem.cnot(i, j)
                new_elem.phase(tpow, j)
                new_elem.cnot(i, j)
                new_circuit.cx(i, j)
                new_circuit.append(U1Gate(tpow * np.pi / 4), [j])
                new_circuit.cx(i, j)

    # Do u1 gates to construct all monomials of weight 1
    for i in range(num_qubits):
        tpow1 = elem_cpy.poly.get_term([i])
        tpow2 = new_elem.poly.get_term([i])
        tpow = (tpow1 - tpow2) % 8
        if tpow != 0:
            new_elem.phase(tpow, i)
            new_circuit.append(U1Gate(tpow * np.pi / 4), [i])

    if elem.poly != new_elem.poly:
        raise QiskitError("Could not recover phase polynomial.")

    inv_circuit = circuit.inverse()
    return new_circuit.combine(inv_circuit)


def random_cnotdihedral(num_qubits, seed=None):
    """Return a random CNOTDihedral element.

    Args:
        num_qubits (int): the number of qubits for the CNOTDihedral object.
        seed (int or RandomState): Optional. Set a fixed seed or
                                   generator for RNG.
    Returns:
        CNOTDihedral: a random CNOTDihedral element.
    """

    if seed is None:
        rng = np.random
    elif isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    elem = CNOTDihedral(num_qubits)

    # Random phase polynomial weights
    weight_1 = rng.randint(8, size=num_qubits)
    elem.poly.weight_1 = weight_1
    weight_2 = 2 * rng.randint(4, size=int(num_qubits * (num_qubits - 1) / 2))
    elem.poly.weight_2 = weight_2
    weight_3 = 4 * rng.randint(2, size=int(num_qubits * (num_qubits - 1) *
                                           (num_qubits - 2) / 6))
    elem.poly.weight_3 = weight_3

    # Random affine function
    # Random invertible binary matrix
    det = 0
    while np.allclose(det, 0) or np.allclose(det, 2):
        linear = rng.randint(2, size=(num_qubits, num_qubits))
        det = np.linalg.det(linear) % 2
    elem.linear = linear

    # Random shift
    shift = rng.randint(2, size=num_qubits)
    elem.shift = shift

    return elem
