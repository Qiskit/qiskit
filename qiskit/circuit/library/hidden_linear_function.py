# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member

"""Hidden Linear Function circuit."""

from typing import Union, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class HiddenLinearFunction(QuantumCircuit):
    r"""Circuit to solve the hidden linear function problem.

    The 2D Hidden Linear Function problem is determined by a 2D adjacency
    matrix A, where only elements that are nearest-neighbor on a grid have
    non-zero entries. Each row/column corresponds to one binary variable
    :math:`x_i`.

    The hidden linear function problem is as follows:

    Consider the quadratic form

    .. math::

        q(x) = \sum_{i,j=1}^{n}{x_i x_j} ~(\mathrm{mod}~ 4)

    and restrict :math:`q(x)` onto the nullspace of A. This results in a linear
    function.

    .. math::

        2 \sum_{i=1}^{n}{z_i x_i} ~(\mathrm{mod}~ 4)  \forall  x \in \mathrm{Ker}(A)

    and the goal is to recover this linear function (equivalently a vector
    :math:`[z_0, ..., z_{n-1}]`). There can be multiple solutions.

    In [1] it is shown that the present circuit solves this problem
    on a quantum computer in constant depth, whereas any corresponding
    solution on a classical computer would require circuits that grow
    logarithmically with :math:`n`. Thus this circuit is an example
    of quantum advantage with shallow circuits.

    **Reference Circuit:**

        .. jupyter-execute::
            :hide-code:

            from qiskit.circuit.library import HiddenLinearFunction
            import qiskit.tools.jupyter
            A = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
            circuit = HiddenLinearFunction(A)
            %circuit_library_info circuit


    **Reference:**

    [1] S. Bravyi, D. Gosset, R. Koenig, Quantum Advantage with Shallow Circuits, 2017.
    `arXiv:1704.00690 <https://arxiv.org/abs/1704.00690>`_
    """

    def __init__(self,
                 adjacency_matrix: Union[List[List[int]], np.ndarray]) -> None:
        """Create new HLF circuit.

        Args:
            adjacency_matrix: a symmetric n-by-n list of 0-1 lists.
                n will be the number of qubits.

        Raises:
            CircuitError: If A is not symmetric.
        """
        adjacency_matrix = np.asarray(adjacency_matrix)
        if not np.allclose(adjacency_matrix, adjacency_matrix.transpose()):
            raise CircuitError("The adjacency matrix must be symmetric.")

        num_qubits = len(adjacency_matrix)
        super().__init__(num_qubits,
                         name="hlf: %s" % adjacency_matrix)

        self.h(range(num_qubits))
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if adjacency_matrix[i][j]:
                    self.cz(i, j)
        for i in range(num_qubits):
            if adjacency_matrix[i][i]:
                self.s(i)
        self.h(range(num_qubits))
