# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


class Configuration:
    """Base configuration object

    This class is the abstract class that all backend configuration is based
    on. The properties of the class are intended to be all ddynamically
    adjustable so that a user can reconfigure the backend on demand. If a
    property is immutable to the user (eg something like number of qubits)
    that should be a property of the backend class itself instead of the
    configuration.
    """

    def __init__(self, **kwargs):
        self.data = kwargs if kwargs else {}

    def update_config(self, **fields):
        self.data.update(fields)

    def get(self, field, default=None):
        return self.data.get(field, default)
