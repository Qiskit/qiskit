# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


class QChannelRep:
    """Quantum channel representation base class."""

    def __init__(self, rep, data, input_dim, output_dim):
        if not isinstance(rep, str):
            raise ValueError("rep must be a string not a {}".format(rep.__class__))
        self._rep = rep
        self._data = data
        self._input_dim = input_dim
        self._output_dim = output_dim

    def __eq__(self, other):
        if (isinstance(other, self.__class__) and
                self.shape == other.shape and
                self.data.__eq__(other.data)):
            return True
        else:
            return False

    def __repr__(self):
        return '{}({}, input_dim={}, output_dim={})'.format(self.rep,
                                                            self.data,
                                                            self.input_dim,
                                                            self.output_dim)

    @property
    def rep(self):
        return self._rep

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def shape(self):
        return self.output_dim, self.input_dim

    @property
    def data(self):
        return self._data
