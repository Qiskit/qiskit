# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract QuantumState class.
"""

import copy
import warnings
from abc import abstractmethod

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator, AbstractTolerancesMeta
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts


class QuantumState(metaclass=AbstractTolerancesMeta):
    """Abstract quantum state base class"""

    def __init__(self, dims):
        """Initialize a state object."""
        # Dimension attributes
        # Note that the tuples of input and output dims are ordered
        # from least-significant to most-significant subsystems
        self._dims = None        # tuple of dimensions of each subsystem
        self._dim = None         # combined dimension of all subsystems
        self._num_qubits = None  # number of qubit subsystems if N-qubit state
        self._set_dims(dims)
        # RNG for measure functions
        self._rng_generator = None

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.dims() == other.dims()

    @property
    def dim(self):
        """Return total state dimension."""
        return self._dim

    @property
    def num_qubits(self):
        """Return the number of qubits if a N-qubit state or None otherwise."""
        return self._num_qubits

    @property
    def atol(self):
        """The absolute tolerance parameter for float comparisons."""
        return self.__class__.atol

    @property
    def rtol(self):
        """The relative tolerance parameter for float comparisons."""
        return self.__class__.rtol

    @classmethod
    def set_atol(cls, value):
        """Set the class default absolute tolerance parameter for float comparisons.

        DEPRECATED: use operator.atol = value instead
        """
        warnings.warn("`{}.set_atol` method is deprecated, use `{}.atol = "
                      "value` instead.".format(cls.__name__, cls.__name__),
                      DeprecationWarning)
        cls.atol = value

    @classmethod
    def set_rtol(cls, value):
        """Set the class default relative tolerance parameter for float comparisons.

        DEPRECATED: use operator.rtol = value instead
        """
        warnings.warn("`{}.set_rtol` method is deprecated, use `{}.rtol = "
                      "value` instead.".format(cls.__name__, cls.__name__),
                      DeprecationWarning)
        cls.rtol = value

    @property
    def _rng(self):
        if self._rng_generator is None:
            return np.random
        return self._rng_generator

    def _reshape(self, dims=None):
        """Reshape dimensions of the state.

        Arg:
            dims (tuple): new subsystem dimensions.

        Returns:
            self: returns self with reshaped dimensions.

        Raises:
            QiskitError: if combined size of all subsystem dimensions are not constant.
        """
        if dims is not None:
            if np.product(dims) != self._dim:
                raise QiskitError(
                    "Reshaped dims are incompatible with combined dimension."
                )
            self._dims = tuple(dims)
        return self

    def dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        if qargs is None:
            return self._dims
        return tuple(self._dims[i] for i in qargs)

    def copy(self):
        """Make a copy of current operator."""
        return copy.deepcopy(self)

    def seed(self, value=None):
        """Set the seed for the quantum state RNG."""
        if value is None:
            self._rng_generator = None
        elif isinstance(value, np.random.Generator):
            self._rng_generator = value
        else:
            self._rng_generator = np.random.default_rng(value)

    @abstractmethod
    def is_valid(self, atol=None, rtol=None):
        """Return True if a valid quantum state."""
        pass

    @abstractmethod
    def to_operator(self):
        """Convert state to matrix operator class"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the operator."""
        pass

    @abstractmethod
    def trace(self):
        """Return the trace of the quantum state as a density matrix."""
        pass

    @abstractmethod
    def purity(self):
        """Return the purity of the quantum state."""
        pass

    @abstractmethod
    def tensor(self, other):
        """Return the tensor product state self ⊗ other.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            QuantumState: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        pass

    @abstractmethod
    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            QuantumState: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        pass

    def _add(self, other):
        """Return the linear combination self + other.

        Args:
            other (QuantumState): a state object.

        Returns:
            QuantumState: the linear combination self + other.

        Raises:
            NotImplementedError: if subclass does not support addition.
        """
        raise NotImplementedError(
            "{} does not support addition".format(type(self)))

    def _multiply(self, other):
        """Return the scalar multipled state other * self.

        Args:
            other (complex): a complex number.

        Returns:
            QuantumState: the scalar multipled state other * self.

        Raises:
            NotImplementedError: if subclass does not support scala
                                 multiplication.
        """
        raise NotImplementedError(
            "{} does not support scalar multiplication".format(type(self)))

    def add(self, other):
        """Return the linear combination self + other.

        DEPRECATED: use ``state + other`` instead.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            LinearOperator: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        warnings.warn("`{}.add` method is deprecated, use + binary operator"
                      "`state + other` instead.".format(self.__class__),
                      DeprecationWarning)
        return self._add(other)

    def subtract(self, other):
        """Return the linear operator self - other.

        DEPRECATED: use ``state - other`` instead.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            LinearOperator: the linear combination self - other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        warnings.warn("`{}.subtract` method is deprecated, use - binary operator"
                      "`state - other` instead.".format(self.__class__),
                      DeprecationWarning)
        return self._add(-other)

    def multiply(self, other):
        """Return the scalar multipled state other * self.

        Args:
            other (complex): a complex number.

        Returns:
            QuantumState: the scalar multipled state other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        warnings.warn("`{}.multiply` method is deprecated, use * binary operator"
                      "`other * state` instead.".format(self.__class__),
                      DeprecationWarning)
        return self._multiply(other)

    @abstractmethod
    def evolve(self, other, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            other (Operator or QuantumChannel): The operator to evolve by.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified QuantumState subsystem dimensions.
        """
        pass

    @abstractmethod
    def expectation_value(self, oper, qargs=None):
        """Compute the expectation value of an operator.

        Args:
            oper (BaseOperator): an operator to evaluate expval.
            qargs (None or list): subsystems to apply the operator on.

        Returns:
            complex: the expectation value.
        """
        pass

    @abstractmethod
    def probabilities(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.
        """
        pass

    def probabilities_dict(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability dictionary.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            dict: The measurement probabilities in dict (ket) form.
        """
        return self._vector_to_dict(
            self.probabilities(qargs=qargs, decimals=decimals),
            self.dims(qargs),
            string_labels=True)

    def sample_memory(self, shots, qargs=None):
        """Sample a list of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            np.array: list of sampled counts if the order sampled.

        Additional Information:

            This function *samples* measurement outcomes using the measure
            :meth:`probabilities` for the current state and `qargs`. It does
            not actually implement the measurement so the current state is
            not modified.

            The seed for random number generator used for sampling can be
            set to a fixed value by using the stats :meth:`seed` method.
        """
        # Get measurement probabilities for measured qubits
        probs = self.probabilities(qargs)

        # Generate list of possible outcome string labels
        labels = self._index_to_ket_array(
            np.arange(len(probs)), self.dims(qargs), string_labels=True)
        return self._rng.choice(labels, p=probs, size=shots)

    def sample_counts(self, shots, qargs=None):
        """Sample a dict of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            Counts: sampled counts dictionary.

        Additional Information:

            This function *samples* measurement outcomes using the measure
            :meth:`probabilities` for the current state and `qargs`. It does
            not actually implement the measurement so the current state is
            not modified.

            The seed for random number generator used for sampling can be
            set to a fixed value by using the stats :meth:`seed` method.
        """
        # Sample list of outcomes
        samples = self.sample_memory(shots, qargs=qargs)

        # Combine all samples into a counts dictionary
        inds, counts = np.unique(samples, return_counts=True)
        return Counts(zip(inds, counts))

    def measure(self, qargs=None):
        """Measure subsystems and return outcome and post-measure state.

        Note that this function uses the QuantumStates internal random
        number generator for sampling the measurement outcome. The RNG
        seed can be set using the :meth:`seed` method.

        Args:
            qargs (list or None): subsystems to sample measurements for,
                                  if None sample measurement of all
                                  subsystems (Default: None).

        Returns:
            tuple: the pair ``(outcome, state)`` where ``outcome`` is the
                   measurement outcome string label, and ``state`` is the
                   collapsed post-measurement state for the corresponding
                   outcome.
        """
        # Sample a single measurement outcome from probabilities
        dims = self.dims(qargs)
        probs = self.probabilities(qargs)
        sample = self._rng.choice(len(probs), p=probs, size=1)

        # Format outcome
        outcome = self._index_to_ket_array(
            sample, self.dims(qargs), string_labels=True)[0]

        # Convert to projector for state update
        proj = np.zeros(len(probs), dtype=complex)
        proj[sample] = 1 / np.sqrt(probs[sample])

        # Update state object
        # TODO: implement a more efficient state update method for
        # diagonal matrix multiplication
        ret = self.evolve(
            Operator(np.diag(proj), input_dims=dims, output_dims=dims),
            qargs=qargs)

        return outcome, ret

    @classmethod
    def _automatic_dims(cls, dims, size):
        """Check if input dimension corresponds to qubit subsystems."""
        return BaseOperator._automatic_dims(dims, size)

    def _set_dims(self, dims):
        """Set dimension attribute"""
        # Shape lists the dimension of each subsystem starting from
        # least significant through to most significant.
        self._dims = tuple(dims)
        # The total input and output dimensions are given by the product
        # of all subsystem dimensions
        self._dim = np.product(dims)
        # Check if an N-qubit operator
        if set(self._dims) == set([2]):
            # If so set the number of qubits
            self._num_qubits = len(self._dims)
        else:
            # Otherwise set the number of qubits to None
            self._num_qubits = None

    @staticmethod
    def _index_to_ket_array(inds, dims, string_labels=False):
        """Convert an index array into a ket array.

        Args:
            inds (np.array): an integer index array.
            dims (tuple): a list of subsystem dimensions.
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            np.array: an array of ket strings if string_label=True, otherwise
                      an array of ket lists.
        """
        shifts = [1]
        for dim in dims[:-1]:
            shifts.append(shifts[-1] * dim)
        kets = np.array([(inds // shift) % dim for dim, shift in zip(dims, shifts)])

        if string_labels:
            max_dim = max(dims)
            char_kets = np.asarray(kets, dtype=np.unicode_)
            str_kets = char_kets[0]
            for row in char_kets[1:]:
                if max_dim > 10:
                    str_kets = np.char.add(',', str_kets)
                str_kets = np.char.add(row, str_kets)
            return str_kets.T

        return kets.T

    @staticmethod
    def _vector_to_dict(vec, dims, decimals=None, string_labels=False):
        """Convert a vector to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            vec (array): a Numpy vector array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the vector in dictionary `ket` form.
        """
        # Get indices of non-zero elements
        vals = vec if decimals is None else vec.round(decimals=decimals)
        inds, = vals.nonzero()

        # Convert to ket tuple based on subsystem dimensions
        kets = QuantumState._index_to_ket_array(
            inds, dims, string_labels=string_labels)

        # Make dict of tuples
        if string_labels:
            return dict(zip(kets, vec[inds]))

        return {tuple(ket): val for ket, val in zip(kets, vals[inds])}

    @staticmethod
    def _matrix_to_dict(mat, dims, decimals=None, string_labels=False):
        """Convert a matrix to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            mat (array): a Numpy matrix array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the matrix in dictionary `ket` form.
        """
        # Get indices of non-zero elements
        vals = mat if decimals is None else mat.round(decimals=decimals)
        inds_row, inds_col, = vals.nonzero()

        # Convert to ket tuple based on subsystem dimensions
        bras = QuantumState._index_to_ket_array(
            inds_row, dims, string_labels=string_labels)
        kets = QuantumState._index_to_ket_array(
            inds_col, dims, string_labels=string_labels)

        # Make dict of tuples
        if string_labels:
            return {'{}|{}'.format(ket, bra): val for ket, bra, val in zip(
                kets, bras, vals[inds_row, inds_col])}

        return {(tuple(ket), tuple(bra)): val for ket, bra, val in zip(
            kets, bras, vals[inds_row, inds_col])}

    @staticmethod
    def _accumulate_dims(dims, qargs):
        """Flatten subsystem dimensions for unspecified qargs.

        This has the potential to reduce the number of subsystems
        by combining consecutive subsystems between the specified
        qargs. For example, if we had a 5-qubit system with
        ``dims = (2, 2, 2, 2, 2)``, and ``qargs=[0, 4]``, then the
        flattened system will have dimensions ``new_dims = (2, 8, 2)``
        and qargs ``new_qargs = [0, 2]``.

        Args:
            dims (tuple): subsystem dimensions.
            qargs (list): qargs list.

        Returns:
            tuple: the pair (new_dims, new_qargs).
        """

        qargs_map = {}
        new_dims = []

        # Accumulate subsystems that can be combined
        accum = []
        for i, dim in enumerate(dims):
            if i in qargs:
                if accum:
                    new_dims.append(np.product(accum))
                    accum = []
                new_dims.append(dim)
                qargs_map[i] = len(new_dims) - 1
            else:
                accum.append(dim)
        if accum:
            new_dims.append(np.product(accum))
        return tuple(new_dims), [qargs_map[i] for i in qargs]

    @staticmethod
    def _subsystem_probabilities(probs, dims, qargs=None):
        """Marginalize a probability vector according to subsystems.

        Args:
            probs (np.array): a probability vector Numpy array.
            dims (tuple): subsystem dimensions.
            qargs (None or list): a list of subsystems to return
                marginalized probabilities for. If None return all
                probabilities (Default: None).

        Returns:
            np.array: the marginalized probability vector flattened
                      for the specified qargs.
        """

        if qargs is None:
            return probs

        # Accumulate dimensions to trace over
        accum_dims, accum_qargs = QuantumState._accumulate_dims(
            dims, qargs)

        # Get sum axis for maginalized subsystems
        n_qargs = len(accum_dims)
        axis = list(range(n_qargs))
        for i in accum_qargs:
            axis.remove(n_qargs - 1 - i)

        # Reshape the probability to a tensor and sum over maginalized axes
        new_probs = np.sum(np.reshape(probs, list(reversed(accum_dims))),
                           axis=tuple(axis))

        # Transpose output probs based on order of qargs
        if sorted(accum_qargs) != accum_qargs:
            axes = np.argsort(accum_qargs)
            return np.ravel(np.transpose(new_probs, axes=axes))

        return np.ravel(new_probs)

    # Overloads
    def __matmul__(self, other):
        # Check for subsystem case return by __call__ method
        if isinstance(other, tuple) and len(other) == 2:
            return self.evolve(other[0], qargs=other[1])
        return self.evolve(other)

    def __xor__(self, other):
        return self.tensor(other)

    def __mul__(self, other):
        return self._multiply(other)

    def __truediv__(self, other):
        return self._multiply(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self._add(other)

    def __sub__(self, other):
        return self._add(-other)

    def __neg__(self):
        return self._multiply(-1)
