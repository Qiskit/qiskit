# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-param-doc

"""
Measurement options.
"""


class MeasOpts:
    """Measurement options."""

    def __init__(self, name=None, **params):
        """Create new measurement options.

        Parameters:
            name (str): Name of measurement option to be used.
        """
        self.name = name
        self.params = params

    def __eq__(self, other):
        """Two measurement options are the same if they are of the same type
        and have the same name and params.

        Args:
            other (MeasOpts): Other Discriminator/Kernel.

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.name == other.name and \
                self.params == other.params:
            return True
        return False
