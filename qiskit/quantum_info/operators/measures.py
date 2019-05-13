# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
A collection of useful quantum information functions for operators.
"""

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators import SuperOp


def process_fidelity(channel1, channel2, require_cptp=True):
    """Return the process fidelity between two quantum channels.

    This is given by

        F_p(E1, E2) = Tr[S2^dagger.S1])/dim^2

    where S1 and S2 are the SuperOp matrices for channels E1 and E2,
    and dim is the dimension of the input output statespace.

    Args:
        channel1 (QuantumChannel or matrix): a quantum channel or unitary matrix.
        channel2 (QuantumChannel or matrix): a quantum channel or unitary matrix.
        require_cptp (bool): require input channels to be CPTP [Default: True].

    Returns:
        array_like: The state fidelity F(state1, state2).

    Raises:
        QiskitError: if inputs channels do not have the same dimensions,
        have different input and output dimensions, or are not CPTP with
        `require_cptp=True`.
    """
    # First we must determine if input is to be interpreted as a unitary matrix
    # or as a channel.
    # If input is a raw numpy array we will interpret it as a unitary matrix.
    is_cptp1 = None
    is_cptp2 = None
    if isinstance(channel1, (list, np.ndarray)):
        channel1 = Operator(channel1)
        if require_cptp:
            is_cptp1 = channel1.is_unitary()
    if isinstance(channel2, (list, np.ndarray)):
        channel2 = Operator(channel2)
        if require_cptp:
            is_cptp2 = channel2.is_unitary()

    # Next we convert inputs SuperOp objects
    # This works for objects that also have a `to_operator` or `to_channel` method
    s1 = SuperOp(channel1)
    s2 = SuperOp(channel2)

    # Check inputs are CPTP
    if require_cptp:
        # Only check SuperOp if we didn't already check unitary inputs
        if is_cptp1 is None:
            is_cptp1 = s1.is_cptp()
        if not is_cptp1:
            raise QiskitError('channel1 is not CPTP')
        if is_cptp2 is None:
            is_cptp2 = s2.is_cptp()
        if not is_cptp2:
            raise QiskitError('channel2 is not CPTP')

    # Check dimensions match
    input_dim1, output_dim1 = s1.dim
    input_dim2, output_dim2 = s2.dim
    if input_dim1 != output_dim1 or input_dim2 != output_dim2:
        raise QiskitError('Input channels must have same size input and output dimensions.')
    if input_dim1 != input_dim2:
        raise QiskitError('Input channels have different dimensions.')

    # Compute process fidelity
    fidelity = np.trace(s1.compose(s2.adjoint()).data) / (input_dim1 ** 2)
    return fidelity
