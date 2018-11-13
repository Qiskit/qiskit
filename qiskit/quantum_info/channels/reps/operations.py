# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import logging
from numbers import Number
import numpy as np
from scipy.linalg import eigh, svd

from .utils import reshuffle, col_to_basis_matrix, pauli_basis
from .baserep import QChannelRep
from .superop import SuperOp
from .choi import Choi
from .kraus import Kraus
from .ptm import PauliTM
from .chi import Chi
from .stinespring import Stinespring

# Create logger
logger = logging.getLogger(__name__)


def transform_rep(channel, rep):
    """Transform between quantum channel representations.

    Args:
        channel (QChannelRep): a quantum channel
        rep (QChannelRep): QChannelRep class or string.

    Returns:
        A quantum channel in representation rep.

    Raises:
        TypeError: if channel is not a QChannelRep
    """
    if rep == SuperOp or rep == 'SuperOp':
        return _transform_to_superop(channel)
    if rep == Choi or rep == 'Choi':
        return _transform_to_choi(channel)
    if rep == Kraus or rep == 'Kraus':
        return _transform_to_kraus(channel)
    if rep == Chi or rep == 'Chi':
        return _transform_to_chi(channel)
    if rep == PauliTM or rep == 'PauliTM':
        return _transform_to_ptm(channel)
    if rep == Stinespring or rep == 'Stinespring':
        return _transform_to_stinespring(channel)
    raise TypeError("Invalid quantum channel representation: '{}".format(rep))


def evolve_state(channel, rho):
    """Evolve a density matrix by a quantum channel.

    This will convert the input channel to a SuperOp representation
    to compute the evolution.

    Args:
        channel (QChannelRep): quantum channel
        rho (matrix_like): density matrix or vectorized density matrix

    Returns:
        the output density matrix or vectorized density matrix.

    Raises:
        TypeError: if channel is not a QChannel objects
        ValueError: if rho is incorrect dimension
    """
    # Check channel is a valid channel representation
    if not issubclass(channel.__class__, QChannelRep):
        raise TypeError("Input is not a valid channel rep class.")

    # Convert input to SuperOp if it isn't already
    if not isinstance(channel, SuperOp):
        channel = _transform_to_superop(channel)

    # Compute evolution
    input_dim = channel.input_dim
    output_dim = channel.output_dim
    state = np.array(rho, dtype=complex)
    shape = state.shape

    # Check if input is a density-matrix:
    if shape == (input_dim, input_dim):
        # Density matrix evolution
        output = channel.data @ state.reshape(input_dim * input_dim, order='F')
        return output.reshape((output_dim, output_dim), order='F')

    # Check if input is a col-vectorized density matrix
    if shape == (input_dim ** 2,) or shape == (input_dim ** 2, 1):
        # Vectorized density matrix evolution
        return channel.data @ state

    # Check if input is a state-vector and show warning
    if state.shape == (input_dim,) or shape == (input_dim, 1):
        # Input is a statevector
        logger.warning("Input state is a statevector: " +
                       "use `projector(state)` to conver to density matrix.")
    raise ValueError('Input state is incorrect dimension.')


def transpose_channel(a):
    """Return the transpose channel

    Args:
        a (QChannelRep): a quantum channel

    Return:
        QChannelRep: the transpose channel of a.
    """
    # Swaps input and output dimensions
    if hasattr(a, 'transpose_channel'):
        return a.transpose_channel()
    else:
        original_rep = a.rep
        transpose_superop = _transform_to_superop(a).transpose_channel()
        return transform_rep(transpose_superop, original_rep)


def conjugate_channel(a):
    """Return the conjugate channel

    Args:
        a (QChannelRep): a quantum channel

    Return:
        QChannelRep: the conjugate channel of a.
    """
    # Swaps input and output dimensions
    if hasattr(a, 'conjugate_channel'):
        return a.conjugate_channel()
    else:
        original_rep = a.rep
        conjugate_superop = _transform_to_superop(a).conjugate_channel()
        return transform_rep(conjugate_superop, original_rep)


def adjoint_channel(a):
    """Return the adjoint channel

    Args:
        a (QChannelRep): a quantum channel

    Return:
        QChannelRep: the adjoint channel of a.
    """
    # Swaps input and output dimensions
    if hasattr(a, 'adjoint_channel'):
        return a.adjoint_channel()
    else:
        original_rep = a.rep
        adjoint_superop = _transform_to_superop(a).adjoint_channel()
        return transform_rep(adjoint_superop, original_rep)


def compose(a, b):
    """Return the composition channel A.B

    Args:
        a (QChannelRep): channel A
        b (QChannelRep): channel B

    Returns:
        QChannelRep: The composition channel A(B(rho))

    Raises:
        TypeError: if a or b are not QChannel objects
    """
    if not issubclass(a.__class__, QChannelRep) or not issubclass(b.__class__, QChannelRep):
        raise TypeError('Input are not quantum channel reps.')
    if hasattr(a, 'compose'):
        return a.compose(transform_rep(b, a.rep))

    return _transform_to_superop(a).compose(_transform_to_superop(b))


def kron(a, b):
    """Return the composite channel kron(A, B)

    Args:
        a (QChannelRep): channel A
        b (QChannelRep): channel B

    Returns:
        QChannelRep: for composite channel kron(A, B)

    Raises:
        TypeError: if a or b are not QChannelRep subclasses
    """
    if not issubclass(a.__class__, QChannelRep) or not issubclass(b.__class__, QChannelRep):
        raise TypeError('Input are not quantum channel reps.')

    # If channels are same representation use internal kron function
    if a.__class__ == b.__class__:
        return a.kron(b)
    # If b is SuperOp preference that either is SuperOp use that convention.
    if b.rep in ['SuperOp', 'Choi'] and a.__class__ != SuperOp:
        return transform_rep(a, b.__class__).kron(b)
    # Finally use rep of channel a
    return a.kron(transform_rep(b, a.__class__))


def add(a, b):
    """Return the channel A + B

    Args:
        a (QChannelRep): channel A
        b (QChannelRep): channel B

    Returns:
        QChannelRep: The composition channel A + B

    Raises:
        TypeError: if a is not a QChannelRep subclass
    """
    if not issubclass(a.__class__, QChannelRep) or not issubclass(b.__class__, QChannelRep):
        raise TypeError('Input are not quantum channel reps.')
    if a.shape != b.shape:
        raise ValueError('Shape of channels do not match.')
    # If a has an add method return in same representation as a.
    if hasattr(a, '__add__'):
        return a.__add__(transform_rep(b, a.rep))
    # Otherwise convert to superoperators and return SuperOp
    if b.rep in ['SuperOp', 'Choi'] and a.__class__ != SuperOp:
        return transform_rep(a, b.__class__).__add__(b)
    return _transform_to_superop(a).__add__(_transform_to_superop(b))


def subtract(a, b):
    """Return the channel A - B

    Args:
        a (QChannelRep): channel A
        b (QChannelRep): channel B

    Returns:
        QChannelRep: The composition channel A - B

    Raises:
        TypeError: if a is not a QChannelRep subclass
    """
    if not issubclass(a.__class__, QChannelRep) or not issubclass(b.__class__, QChannelRep):
        raise TypeError('Input are not quantum channel reps.')
    if a.shape != b.shape:
        raise ValueError('Shape of channels do not match.')
    # If a has an sub method return in same representation as a.
    if hasattr(a, '__sub__'):
        return a.__sub__(transform_rep(b, a.rep))
    # Otherwise convert to superoperators and return SuperOp
    if b.rep in ['SuperOp', 'Choi'] and a.__class__ != SuperOp:
        return transform_rep(a, b.__class__).__sub__(b)
    return _transform_to_superop(a).__sub__(_transform_to_superop(b))


def multiply(a, num):
    """Return the channel number * A

    Args:
        a (QChannelRep): channel A
        num (Number): a scalar number

    Returns:
        QChannelRep: The channel num * A

    Raises:
        TypeError: if a is not a QChannelRep subclass
    """
    if not issubclass(a.__class__, QChannelRep):
        raise TypeError('Input a is not quantum channel reps.')
    if not hasattr(a, "__mul__"):
        a = _transform_to_superop(a)
    return a.__mul__(num)


def power(a, power):
    """Return the composition channel A^power

    Args:
        a (QChannelRep): channel A
        power (int): a positive integer power

    Returns:
        QChannelRep: The channel A.A....A (power times)

    Raises:
        TypeError: if a is not a QChannelRep subclass
    """
    if not issubclass(a.__class__, QChannelRep):
        raise TypeError('Input a is not quantum channel reps.')
    if not hasattr(a, "__pow__"):
        a = _transform_to_superop(a)
    return a.__pow__(power)


def negate(a):
    """Return the channel -A

    Args:
        a (QChannelRep): channel A

    Returns:
        QChannelRep: The composition channel A(B(rho))

    Raises:
        TypeError: if a is not a QChannelRep subclass
    """
    if not issubclass(a.__class__, QChannelRep):
        raise TypeError('Input are not quantum channel reps.')
    if not hasattr(a, '__neg__'):
        a = _transform_to_superop(a)
    return a.__neg__()


def _transform_to_superop(channel):
    """Transform a channel to the SuperOp representation."""
    if isinstance(channel, SuperOp):
        return channel
    else:
        din = channel.input_dim
        dout = channel.output_dim

        # Transform from Choi
        if isinstance(channel, Choi):
            data = reshuffle(channel.data, shape=(din, dout, din, dout))
            return SuperOp(data, input_dim=din, output_dim=dout)

        # Transform from Kraus
        if isinstance(channel, Kraus):
            data = 0
            kraus_l, kraus_r = channel._data
            if kraus_r is None:
                for a in kraus_l:
                    data += np.kron(np.conj(a), a)
            else:
                for a, b in zip(kraus_l, kraus_r):
                    data += np.kron(np.conj(b), a)
            return SuperOp(data, input_dim=din, output_dim=dout)

        # Transform from Stinespring
        if isinstance(channel, Stinespring):
            return _transform_to_superop(_transform_to_kraus(channel))

        # Transfrom from Pauli transfer matrix
        if isinstance(channel, PauliTM):
            num_qubits = int(np.log2(din))
            # Change basis
            # Note that we manually renormalized the change of basis matrix
            # to avoid rounding errors from square-roots of 2.
            cobmat = col_to_basis_matrix(pauli_basis(num_qubits))
            norm = 2 ** num_qubits
            return SuperOp(cobmat.T.conj().dot(channel.data).dot(cobmat) / norm,
                           input_dim=din, output_dim=dout)

        # Transfrom from Pauli Chi-matrix
        if isinstance(channel, Chi):
            return _transform_to_superop(_transform_to_choi(channel))

        # Transformation failed
        raise TypeError(_transform_fail_message(channel, SuperOp))


def _transform_to_choi(channel):
    """Transform a channel to the Choi representation."""
    if isinstance(channel, Choi):
        return channel
    else:
        din = channel.input_dim
        dout = channel.output_dim

        # Transform from SuperOp
        if isinstance(channel, SuperOp):
            data = reshuffle(channel.data, (dout, dout, din, din))
            return Choi(data, input_dim=din, output_dim=dout)

        # Transform from Kraus
        if isinstance(channel, Kraus):
            data = 0
            kraus_l, kraus_r = channel._data
            if kraus_r is None:
                for a in kraus_l:
                    u = a.ravel(order='F')
                    data += np.dot(u[:, None], u[None, :])
            else:
                for a, b in zip(kraus_l, kraus_r):
                    data += np.dot(a.ravel(order='F')[:, None],
                                   b.ravel(order='F').conj()[None, :])
            return Choi(data, input_dim=din, output_dim=dout)

        # Transform from Stinespring
        if isinstance(channel, Stinespring):
            return _transform_to_choi(_transform_to_kraus(channel))

        # Transfrom from Pauli transfer matrix
        if isinstance(channel, PauliTM):
            return _transform_to_choi(_transform_to_superop(channel))

        # Transform from Chi matrix
        if isinstance(channel, Chi):
            num_qubits = int(np.log2(din))
            # Change basis
            # Note that we manually renormalized the change of basis matrix
            # to avoid rounding errors from square-roots of 2.
            cobmat = col_to_basis_matrix(pauli_basis(num_qubits))
            norm = 2 ** num_qubits
            return Choi(cobmat.T.conj().dot(channel.data).dot(cobmat) / norm,
                        input_dim=din, output_dim=dout)

        # Transformation failed
        raise TypeError(_transform_fail_message(channel, Choi))


def _transform_to_kraus(channel, threshold=1e-10):
    """Transform a channel to the Kraus representation."""
    if isinstance(channel, Kraus):
        return channel
    else:

        # Transform from Choi
        if isinstance(channel, Choi):
            din = channel.input_dim
            dout = channel.output_dim
            # Get eigen-decomposition of Choi-matrix
            w, v = eigh(channel.data)

            # Check eigenvaleus are positive
            if len(w[w < -threshold]) == 0:
                # CP-map Kraus representation
                kraus = []
                for val, vec in zip(w, v.T):
                    if abs(val) > threshold:
                        k = np.sqrt(val) * vec.reshape((dout, din), order='F')
                        kraus.append(k)
                return Kraus((kraus, None), input_dim=din, output_dim=dout)
            else:
                # Non-CP-map generalized Kraus representation
                U, s, Vh = svd(channel.data)
                kraus_l = []
                kraus_r = []
                for val, vecL, vecR in zip(s, U.T, Vh.conj()):
                    kraus_l.append(np.sqrt(val) * vecL.reshape((dout, din), order='F'))
                    kraus_r.append(np.sqrt(val) * vecR.reshape((dout, din), order='F'))
                return Kraus((kraus_l, kraus_r), input_dim=din, output_dim=dout)

        # Transform from Stinespring
        if isinstance(channel, Stinespring):
            din = channel.input_dim
            dout = channel.output_dim
            kraus_pair = []
            for stine in channel._data:
                if stine is None:
                    kraus_pair.append(None)
                else:
                    trace_dim = stine.shape[0] // dout
                    iden = np.eye(dout)
                    kraus = []
                    for j in range(trace_dim):
                        v = np.zeros(trace_dim)
                        v[j] = 1
                        kraus.append(np.kron(iden, v[None, :]).dot(stine))
                    kraus_pair.append(kraus)
            Kraus(tuple(kraus_pair), input_dim=din, output_dim=dout)

        # For other representations transform to Choi first
        return _transform_to_kraus(_transform_to_choi(channel))


def _transform_to_stinespring(channel):
    """Transform a channel to the Stinespring representation."""
    if isinstance(channel, Stinespring):
        return channel

    # Transform from Kraus
    elif isinstance(channel, Kraus):
        din = channel.input_dim
        dout = channel.output_dim
        stine_pair = []
        for kraus in channel._data:
            if kraus is None:
                stine_pair.append(None)
            else:
                num_kraus = len(kraus)
                stine = np.zeros((dout * num_kraus, din), dtype=complex)
                for j, k in enumerate(kraus):
                    v = np.zeros(num_kraus)
                    v[j] == 1
                    stine += np.kron(k, v[:, None])
                stine_pair.append(stine)
        return Stinespring(tuple(stine_pair), input_dim=din, output_dim=dout)

    # For other representations transform to Kraus first
    return _transform_to_stinespring(_transform_to_kraus(channel))


def _transform_to_ptm(channel):
    """Transform a channel to the Pauli transfer matrix representation."""
    if isinstance(channel, PauliTM):
        return channel
    elif isinstance(channel, SuperOp):
        din = channel.input_dim
        dout = channel.output_dim
        if din != dout:
            raise ValueError("Invalid input and output dimensions must be " +
                             "equal for Pauli transfer matrix.")
        num_qubits = int(np.log2(din))
        if 2 ** num_qubits != din:
            raise ValueError("Input is not an n-qubit channel.")
        # Change basis
        # Note that we manually renormalized the change of basis matrix
        # to avoid rounding errors from square-roots of 2.
        cobmat = col_to_basis_matrix(pauli_basis(num_qubits))
        norm = 2 ** num_qubits
        return PauliTM(cobmat.dot(channel.data).dot(cobmat.T.conj()) / norm,
                       input_dim=din, output_dim=dout)
    # For other representations transform to SuperOp first
    return _transform_to_ptm(_transform_to_superop(channel))


def _transform_to_chi(channel):
    """Transform a channel to the Chi matrix representation."""
    if isinstance(channel, Chi):
        return channel
    elif isinstance(channel, Choi):
        din = channel.input_dim
        dout = channel.output_dim
        if din != dout:
            raise ValueError("Invalid input and output dimensions " +
                             "must be equal for Chi matrix.")
        num_qubits = int(np.log2(din))
        if 2 ** num_qubits != din:
            raise ValueError("Input is not an n-qubit channel.")
        # Change basis
        # Note that we manually renormalized the change of basis matrix
        # to avoid rounding errors from square-roots of 2.
        cobmat = col_to_basis_matrix(pauli_basis(num_qubits))
        norm = 2 ** num_qubits
        return Chi(cobmat.dot(channel.data).dot(cobmat.T.conj()) / norm,
                   input_dim=din, output_dim=dout)
    # Transform via Choi rep
    return _transform_to_chi(_transform_to_choi(channel))


def _transform_fail_message(channel, target_class):
    return "Unable to transform channel rep {} to {}".format(channel.rep, target_class)
