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

"""Methods for generating random quantum information objects."""

# pylint: disable=unused-import
from qiskit.quantum_info.operators.random import (random_unitary,
                                                  random_quantum_channel,
                                                  random_hermitian,
                                                  random_clifford,
                                                  random_pauli_table,
                                                  random_stabilizer_table)

from qiskit.quantum_info.states.random import (random_statevector,
                                               random_density_matrix)

from qiskit.quantum_info.states.random import random_state  # DEPRECATED in 0.13
