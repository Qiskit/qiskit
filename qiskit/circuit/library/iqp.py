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

"""Instantaneous quantum polynomial circuit."""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func


class IQP(QuantumCircuit):
    r"""Instantaneous quantum polynomial (IQP) circuit.

    The circuit consists of a column of Hadamard gates,
    a column of powers of T gates,
    a sequence of powers of CS gates (up to
    :math:`\frac{n^2-n}{2}` of them),
    and a final column of Hadamard gates, as introduced in [1].

    The circuit is parameterized by an n x n interactions matrix.
    The powers of each T gate are given by the diagonal elements
    of the interactions matrix. The powers of the CS gates are
    given by the upper triangle of the interactions matrix.

    **Reference Circuit:**

    .. plot::

       from qiskit.circuit.library import IQP
       A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
       circuit = IQP(A)
       circuit.draw('mpl')

    **Expanded Circuit:**

        .. plot::

           from qiskit.circuit.library import IQP
           from qiskit.visualization.library import _generate_circuit_library_visualization
           A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
           circuit = IQP(A)
           _generate_circuit_library_visualization(circuit.decompose())

    **References:**

    [1] M. J. Bremner et al. Average-case complexity versus approximate
    simulation of commuting quantum computations,
    Phys. Rev. Lett. 117, 080501 (2016).
    `arXiv:1504.07999 <https://arxiv.org/abs/1504.07999>`_
    """

    @deprecate_func(
        since="1.3",
        additional_msg="Use the qiskit.circuit.library.iqp function instead.",
        pending=True,
    )
    def __init__(self, interactions: list | np.ndarray) -> None:
        """Create IQP circuit.

        Args:
            interactions: input n-by-n symmetric matrix.

        Raises:
            CircuitError: if the inputs is not as symmetric matrix.
        """
        circuit = iqp(interactions)
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


def iqp(
    interactions: Sequence[Sequence[int]] | None = None, num_qubits: int | None = None
) -> QuantumCircuit:
    r"""Instantaneous quantum polynomial (IQP) circuit.

    The circuit consists of a column of Hadamard gates, a column of powers of T gates,
    a sequence of powers of CS gates (up to :math:`\frac{n^2-n}{2}` of them), and a final column of
    Hadamard gates, as introduced in [1].

    The circuit is parameterized by an :math:`n \times n` interactions matrix. The powers of each
    T gate are given by the diagonal elements of the interactions matrix. The powers of the CS gates
    are given by the upper triangle of the interactions matrix.

    **Reference Circuit:**

    .. plot::

       from qiskit.circuit.library import iqp
       A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
       circuit = iqp(A)
       circuit.draw("mpl")

    **Expanded Circuit:**

        .. plot::

           from qiskit.circuit.library import iqp
           from qiskit.visualization.library import _generate_circuit_library_visualization
           A = [[6, 5, 3], [5, 4, 5], [3, 5, 1]]
           circuit = iqp(A)
           _generate_circuit_library_visualization(circuit)

    **References:**

    [1] M. J. Bremner et al. Average-case complexity versus approximate
    simulation of commuting quantum computations,
    Phys. Rev. Lett. 117, 080501 (2016).
    `arXiv:1504.07999 <https://arxiv.org/abs/1504.07999>`_

    Args:
        interactions: The interactions as symmetric square matrix of width ``num_qubits``.
        num_qubits: If no interactions are given, construct a random IQP circuit with this
            number of qubits. This argument cannot be passed at the same time as ``interactions``.

    Returns:
        An IQP circuit.
    """
    # if no interactions are given, generate them
    if interactions is None:
        if num_qubits is None:
            raise ValueError("Either interactions or num_qubits must be provided.")

        # create a symmetric random interactions matrix
        random_full = np.random.random_integers(0, 8, size=(num_qubits, num_qubits))
        interactions = np.triu(random_full) + np.triu(random_full, -1)

    # otherwise validate the interactions
    else:
        if num_qubits is not None:
            raise ValueError("Only one of interactions or num_qubits can be provided, not both.")

        num_qubits = len(interactions)
        interactions = np.asarray(interactions)

        if not np.allclose(interactions, interactions.T):
            raise CircuitError("The interactions matrix is not symmetric")

    # set the label -- if the number of qubits is too large, do not show the interactions matrix
    if num_qubits < 5:
        label = np.array_str(interactions)
        name = "iqp:" + label.replace("\n", ";")
    else:
        label = "iqp"

    # construct the circuit
    circuit = QuantumCircuit(num_qubits, name=name)

    circuit.h(range(num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if interactions[i][j] % 4 != 0:
                circuit.cp(interactions[i][j] * np.pi / 2, i, j)

    for i in range(num_qubits):
        if interactions[i][i] % 8 != 0:
            circuit.p(interactions[i][i] * np.pi / 8, i)

    circuit.h(range(num_qubits))

    return circuit
