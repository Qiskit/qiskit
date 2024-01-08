# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,no-name-in-module,ungrouped-imports

"""A circuit library visualization"""

from qiskit import QuantumCircuit
from qiskit.utils import optionals as _optionals


@_optionals.HAS_MATPLOTLIB.require_in_call
def _generate_circuit_library_visualization(circuit: QuantumCircuit):
    import matplotlib.pyplot as plt

    circuit = circuit.decompose()
    ops = circuit.count_ops()
    num_nl = circuit.num_nonlocal_gates()
    _fig, (ax0, ax1) = plt.subplots(2, 1)
    circuit.draw("mpl", ax=ax0)
    ax1.axis("off")
    ax1.grid(visible=None)
    ax1.table(
        [[circuit.name], [circuit.width()], [circuit.depth()], [sum(ops.values())], [num_nl]],
        rowLabels=["Circuit Name", "Width", "Depth", "Total Gates", "Non-local Gates"],
    )
    plt.tight_layout()
    plt.show()
