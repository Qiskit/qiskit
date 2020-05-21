# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Aqua Globals """

import logging

import numpy as np
from qiskit.util import local_hardware_info
import qiskit

from .aqua_error import AquaError


logger = logging.getLogger(__name__)


class QiskitAquaGlobals:
    """Aqua class for global properties."""

    CPU_COUNT = local_hardware_info()['cpus']

    def __init__(self):
        self._random_seed = None
        self._num_processes = QiskitAquaGlobals.CPU_COUNT
        self._random = None

    @property
    def random_seed(self):
        """Return random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """Set random seed."""
        self._random_seed = seed
        self._random = None

    @property
    def num_processes(self):
        """Return num processes."""
        return self._num_processes

    @num_processes.setter
    def num_processes(self, num_processes):
        """Set num processes."""
        if num_processes < 1:
            raise AquaError('Invalid Number of Processes {}.'.format(num_processes))
        if num_processes > QiskitAquaGlobals.CPU_COUNT:
            raise AquaError('Number of Processes {} cannot be greater than cpu count {}.'
                            .format(num_processes, QiskitAquaGlobals.CPU_COUNT))
        self._num_processes = num_processes
        # TODO: change Terra CPU_COUNT until issue
        # gets resolved: https://github.com/Qiskit/qiskit-terra/issues/1963
        try:
            qiskit.tools.parallel.CPU_COUNT = self.num_processes
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning("Failed to set qiskit.tools.parallel.CPU_COUNT "
                           "to value: '%s': Error: '%s'", self.num_processes, str(ex))

    @property
    def random(self):
        """Return a numpy np.random.Generator (default_rng)."""
        if self._random is None:
            self._random = np.random.default_rng(self._random_seed)
        return self._random


# Global instance to be used as the entry point for globals.
aqua_globals = QiskitAquaGlobals()  # pylint: disable=invalid-name
