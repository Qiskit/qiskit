# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QChannel Class

For explanation of terminology and details of operations see Ref. [1]

References:
    [1] C.J. Wood, J.D. Biamonte Smith, D.G. Cory Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from .reps import operations
from .reps.baserep import QChannelRep
from .reps.superop import SuperOp
from .reps.choi import Choi
from .reps.kraus import Kraus
from .reps.stinespring import Stinespring
from .reps.ptm import PauliTM
from .reps.chi import Chi


class QChannel:
    """Quantum Channel class"""

    _QCHANNELREP_LABELS = {
        SuperOp: 'SuperOp',
        Choi: 'Choi',
        Kraus: 'Kraus',
        Stinespring: 'Stinespring',
        PauliTM: 'PauliTM',
        Chi: 'Chi'
    }

    def __init__(self, data, rep=None, input_dim=None, output_dim=None):
        if issubclass(data.__class__, QChannelRep):
            self._channel = data
        elif rep is None or rep == 'Kraus' or rep is Kraus:
            # Default to Kraus case so it can handle unitary input.
            self._channel = Kraus(data, input_dim=None, output_dim=output_dim)
        elif rep is SuperOp or rep == 'SuperOp':
            self._channel = SuperOp(data, input_dim=None, output_dim=output_dim)
        elif rep is Choi or rep == 'Choi':
            self._channel = Choi(data, input_dim=None, output_dim=output_dim)
        elif rep is Stinespring or rep == 'Stinespring':
            self._channel = Stinespring(data, input_dim=None, output_dim=output_dim)
        elif rep is PauliTM or rep == 'PauliTM':
            self._channel = PauliTM(data, input_dim=None, output_dim=output_dim)
        elif rep is Chi or rep == 'Chi':
            self._channel = Chi(data, input_dim=None, output_dim=output_dim)
        else:
            raise TypeError("Unrecognised channel representation: '{}".format(rep))
        # Initialize empty cache
        self._cache = {}

    def __repr__(self):
        display = "QChannel({}({}, ".format(self.rep, self.data) + \
                  "input_dim={}, output_dim={}))".format(self.input_dim, self.output_dim)
        return display

    @property
    def channel(self):
        return self._channel

    @property
    def data(self):
        return self.channel.data

    @property
    def rep(self):
        return self.channel.rep

    @property
    def input_dim(self):
        return self.channel.input_dim

    @property
    def output_dim(self):
        return self.channel.output_dim

    @property
    def shape(self):
        return self.channel.shape

    def rep_string(self, rep):
        if isinstance(rep, str):
            return rep
        label = self._QCHANNELREP_LABELS.get(rep, None)
        if label is None:
            raise ValueError("Invalid channel rep")
        return label

    def clear_cached(self):
        """
        Clear cached data from representations other than the current one.
        """
        self._cache.clear()

    def transform(self, rep, inplace=False):
        """Transform between quantum channel representations.

        Args:
            rep (QChannelRep): QChannelRep class or string.
            inplace (bool): tranform inplace or return a new QChannel object.

        Returns:
            A quantum channel in representation rep.

        Raises:
            TypeError: if rep is not a string or QChannelRep.
        """
        if isinstance(rep, str):
            output_rep = rep
        elif issubclass(rep, QChannelRep):
            output_rep = self.rep_string(rep)
            if output_rep is None:
                raise ValueError("Invalid channel rep")
        else:
            raise TypeError("rep must be QChannelRep subclass or str label.")
        return self._transform_helper(output_rep, inplace)

    def _transform_helper(self, output_rep, inplace=False):
        """Transform between quantum channel representations.

        Args:
            rep_str (str): QChannelRep string.
            inplace (bool): tranform inplace or return a new QChannel object.

        Returns:
            A quantum channel in representation rep.

        Raises:
            TypeError: if rep is not a string or QChannelRep.
        """
        original_rep = self.rep
        if original_rep == output_rep:
            # Should this be a copy or deepcopy if inplace=False?
            return self
        if inplace:
            self._cache[original_rep] = self.channel
        # Look for cached rep
        if output_rep in self._cache:
            if inplace:
                self._channel = self._cache.pop(output_rep)
                return self
            else:
                # Should this be a copy or deepcopy?
                return QChannel(self._cache[output_rep])
        # Compute the transformation if not already cached
        if inplace:
            self._channel = operations.transform_rep(self.channel, output_rep)
            return self
        else:
            return QChannel(operations.transform_rep(self.channel, output_rep))

    def evolve_state(self, rho):
        """Evolve a density matrix by a quantum channel.

        Args:
            rho (matrix_like): density matrix or vectorized density matrix

        Returns:
            the output density matrix or vectorized density matrix.
        """
        # Check if QChannel object
        original_rep = self.rep
        # Transform to superop
        self._transform_helper('SuperOp', inplace=True)
        output = operations.evolve_state(self.channel, rho)
        # Transform to original representation
        self._transform_helper(original_rep)
        return output

    def transpose_channel(self):
        """Return the transpose channel"""
        original_rep = self.rep
        # If channel doesn't have method transform to superoperator
        if not hasattr(self.channel, 'transpose_channel'):
            self._transform_helper('SuperOp', inplace=True)
        transpose_channel = self.channel.transpose_channel()
        self._transform_helper(original_rep, inplace=True)
        return QChannel(transpose_channel)._transform_helper(original_rep, inplace=True)

    def conjugate_channel(self):
        """Return the conjugate channel"""
        original_rep = self.rep
        # If channel doesn't have method transform to superoperator
        if not hasattr(self.channel, 'conjugate_channel'):
            self._transform_helper('SuperOp', inplace=True)
        conjugate_channel = self.channel.conjugate_channel()
        self._transform_helper(original_rep, inplace=True)
        return QChannel(conjugate_channel)._transform_helper(original_rep, inplace=True)

    def adjoint_channel(self):
        """Return the adjoint channel"""
        original_rep = self.rep
        # If channel doesn't have method transform to superoperator
        if not hasattr(self.channel, 'adjoint_channel'):
            self._transform_helper('SuperOp', inplace=True)
        adjoint_channel = self.channel.adjoint_channel()
        self._transform_helper(original_rep, inplace=True)
        return QChannel(adjoint_channel)._transform_helper(original_rep, inplace=True)

    def compose(self, b):
        """Return the composition channel A.B

        Args:
            b (QChannel or QChannelRep): channel B

        Returns:
        The composition channel A(B(rho))

        Raises:
            TypeError: if channel b is not a QChannel or QChannelRep
            ValueError: if output_dim of b does not match input_dim
        """
        if isinstance(b, QChannel):
            b = b.channel
        if not issubclass(b.__class__, QChannelRep):
            raise TypeError("Not a quantum channel representation.")
        if b.output_dim != self.input_dim:
            raise ValueError('output_dim of other channel does not match input_dim.')
        return QChannel(operations.compose(self.channel, b))

    def kron(self, b):
        """Return the composite channel A \otimes B

        Args:
            b (QChannel or QChannelRep): channel B

        Returns:
            The composite channel A \otimes B

        Raises:
            TypeError: if channel is not a QChannel or QChannelRep
        """
        if isinstance(b, QChannel):
            b = b.channel
        if not issubclass(b.__class__, QChannelRep):
            raise TypeError("Not a quantum channel representation.")
        return QChannel(operations.kron(self.channel, b))

    # Operator overloads
    def __matmul__(self, other):
        """This is shorthand for channel.operations.compose(self, other)."""
        return self.compose(other)

    def __mul__(self, other):
        original_rep = self.rep
        if not hasattr(self.channel, "__mul__"):
            self._transform_helper('SuperOp', inplace=True)
        result = self.channel.__mul__(other)
        self._transform_helper(original_rep, inplace=True)
        return QChannel(result)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        original_rep = self.rep
        if not hasattr(self.channel, "__pow__"):
            self._transform_helper('SuperOp', inplace=True)
        result = self.channel.__pow__(power)
        self._transform_helper(original_rep, inplace=True)
        return QChannel(result)

    def __add__(self, other):
        if isinstance(other, QChannel):
            other = other.channel
        original_rep = self.rep
        if not hasattr(self.channel, "__add__"):
            self._transform_helper('SuperOp', inplace=True)
        result = operations.add(self.channel, other)
        self._transform_helper(original_rep, inplace=True)
        return QChannel(result)

    def __sub__(self, other):
        if isinstance(other, QChannel):
            other = other.channel
        original_rep = self.rep
        if not hasattr(self.channel, "__sub__"):
            self._transform_helper('SuperOp', inplace=True)
        result = operations.subtract(self.channel, other)
        self._transform_helper(original_rep, inplace=True)
        return result

    def __neg__(self):
        original_rep = self.rep
        if not hasattr(self.channel, "__neg__"):
            self._transform_helper('SuperOp', inplace=True)
        result = self.channel.__neg__()
        self._transform_helper(original_rep, inplace=True)
        return QChannel(result)

    # Assignment overloads
    def __imatmul__(self, other):
        if isinstance(other, QChannel):
            other = other.channel
        channel = operations.compose(self.channel, other)
        self.clear_cached()
        self._channel = channel

    def __imul__(self, other):
        original_rep = self.rep
        if not hasattr(self.channel, "__imul__"):
            self._transform_helper('SuperOp', inplace=True)
        self.channel.__imul__(other)
        self.clear_cached()
        self._transform_helper(original_rep, inplace=True)

    def __itruediv__(self, other):
        return self.__imul__(1 / other)

    def __ipow__(self, power):
        original_rep = self.rep
        if not hasattr(self.channel, "__ipow__"):
            self._transform_helper('SuperOp', inplace=True)
        self.channel.__ipow__(power)
        self.clear_cached()
        self._transform_helper(original_rep, inplace=True)

    def __iadd__(self, other):
        if isinstance(other, QChannel):
            other = other.channel
        original_rep = self.rep
        if not hasattr(self.channel, "__iadd__"):
            self._transform_helper('SuperOp', inplace=True)
        self._channel = operations.add(self.channel, other)
        self.clear_cached()
        self._transform_helper(original_rep, inplace=True)

    def __isub__(self, other):
        if isinstance(other, QChannel):
            other = other.channel
        original_rep = self.rep
        if not hasattr(self.channel, "__isub__"):
            self._transform_helper('SuperOp', inplace=True)
        self._channel = operations.subtract(self.channel, other)
        self.clear_cached()
        self._transform_helper(original_rep, inplace=True)
