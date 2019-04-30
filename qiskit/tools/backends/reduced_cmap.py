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

"""Reduced coupling_map
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
from qiskit.qiskiterror import QiskitError


def reduced_coupling_map(backend, mapping):
    """Returns a reduced coupling map that
    corresponds to the subgraph of backend qubits
    selected in the mapping.

    Args:
        backend (BaseBackend): A backend for a physical device.
        mapping (list): A mapping of reduced qubits to device
                        qubits.

    Returns:
        list: A reduced coupling_map for the selected qubits.

    Raises:
        QiskitError: Backend must be a real device, not simulator.
    """
    n_qubits = backend.configuration().n_qubits
    reduced_qubits = len(mapping)
    coupling_map = backend.configuration().coupling_map

    if coupling_map is None:
        raise QiskitError('Backend must have a coupling_map != None.')

    inv_map = [None]*n_qubits
    for idx, val in enumerate(mapping):
        inv_map[val] = idx

    reduced_cmap = []

    for edge in coupling_map:
        if edge[0] in mapping and edge[1] in mapping:
            reduced_cmap.append([inv_map[edge[0]], inv_map[edge[1]]])

    # Verify coupling_map is connected
    rows = np.array([edge[0] for edge in reduced_cmap], dtype=int)
    cols = np.array([edge[1] for edge in reduced_cmap], dtype=int)
    data = np.ones_like(rows)

    mat = sp.coo_matrix((data, (rows, cols)),
                        shape=(reduced_qubits, reduced_qubits)).tocsr()

    if cs.connected_components(mat)[0] != 1:
        raise QiskitError('coupling_map must be connected.')

    return reduced_cmap
