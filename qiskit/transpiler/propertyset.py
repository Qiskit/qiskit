# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" A property set is maintained by the PassManager to keep information
about the current state of the circuit """


class PropertySet(dict):
    """ A default dictionary-like object """

    def __missing__(self, key):
        return None
