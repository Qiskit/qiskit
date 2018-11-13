# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Convenience functions that work with QChannel or QChannelRep objects"""

from .qchannel import QChannel
from .reps.baserep import QChannelRep
from .reps import operations


def transform_channel(channel, rep):
    """Transform between quantum channel representations.

    Args:
        channel (QChannel or QChannelRep): quantum channel
        rep (QChannelRep): QChannelRep class or string.

    Returns:
        A quantum channel in representation rep.

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(channel.__class__, QChannelRep):
        channel = QChannel(channel)
    if isinstance(channel, QChannel):
        return channel.transform(rep)
    raise TypeError("Invalid input quantum channel.")


def evolve_state(channel, rho):
    """Evolve a density matrix by a quantum channel.

    This will convert the input channel to a SuperOp representation
    to compute the evolution.

    Args:
        channel (QChannel or QChannelRep): quantum channel
        rho (matrix_like): density matrix or vectorized density matrix

    Returns:
        the output density matrix or vectorized density matrix.

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    # Check if QChannel object
    if isinstance(channel, QChannel):
        channel = channel.channel
    if issubclass(channel.__class__, QChannelRep):
        return operations.evolve_state(channel, rho)
    raise TypeError("Invalid input quantum channel.")


def transpose_channel(channel):
    """Return the transpose channel

    Args:
        channel (QChannel or QChannelRep): a quantum channel

    Return:
        QChannel: The transpose of the input channel.

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(channel.__class__, QChannelRep):
        channel = QChannel(channel)
    if isinstance(channel, QChannel):
        return channel.transpose_channel()
    raise TypeError("Invalid input quantum channel.")


def conjugate_channel(channel):
    """Return the conjugate channel

    Args:
        channel (QChannel or QChannelRep): a quantum channel

    Return:
        QChannel: The conjugate of the inputchannel.

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(channel.__class__, QChannelRep):
        channel = QChannel(channel)
    if isinstance(channel, QChannel):
        return channel.conjugate_channel()
    raise TypeError("Invalid input quantum channel.")


def adjoint_channel(channel):
    """Return the adjoint channel

    Args:
        channel (QChannel or QChannelRep): a quantum channel

    Return:
        QChannel: The adjoint of the inputchannel.

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(channel.__class__, QChannelRep):
        channel = QChannel(channel)
    if isinstance(channel, QChannel):
        return channel.adjoint_channel()
    raise TypeError("Invalid input quantum channel.")


def compose(a, b):
    """Return the composition channel A.B

    Args:
        a (QChannel or QChannelRep): channel A
        b (QChannel or QChannelRep): channel B

    Returns:
       QChannel: The composition channel A(B(rho))

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(a.__class__, QChannelRep):
        a = QChannel(a)
    if isinstance(a, QChannel):
        return a.compose(b)
    raise TypeError("Invalid input quantum channel.")


def kron(a, b):
    """Return the composite channel kron(A, B)

    Args:
        a (QChannel or QChannelRep): channel A
        b (QChannel or QChannelRep): channel B

    Returns:
        QChannel: The composite channel kron(A, B)

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(a.__class__, QChannelRep):
        a = QChannel(a)
    if isinstance(a, QChannel):
        return a.kron(b)
    raise TypeError("Invalid input quantum channel.")


def power(a, power):
    """Return the composition channel A^power

    Args:
        a (QChannel or QChannelRep): channel A
        power (int): a positive integer power

    Returns:
        QChannelRep: The channel A.A....A (power times)

    Raises:
        TypeError: if channel is not a QChannel or QChannelRep
    """
    if issubclass(a.__class__, QChannelRep):
        a = QChannel(a)
    if isinstance(a, QChannel):
        return a.__pow__(power)
    raise TypeError("Invalid input quantum channel.")
