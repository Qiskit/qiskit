# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""This package contains Trotterization-based Quantum Real Time Evolution algorithm.
It is compliant with the new Quantum Time Evolution Framework and makes use ProductFormula and
PauliEvolutionGate implementations.
The evolution with gradients assumes that a Hamiltonian is a linear combination of PauliOp objects
w.r.t. given parameters. It case of a single summand, it might be a PauliOp, or an OperatorBase.
Gradients are taken w.r.t. to a time parameter using the finite difference method (if
a Hamiltonian is time-dependent via t_param = Parameter("t")) and/or w.r.t. parameters present in a
Hamiltonian using an expectation value through a custom observable."""

from qiskit.algorithms.quantum_time_evolution.real.implementations.trotterization.trotter_qrte import (
    TrotterQrte,
)

__all__ = ["TrotterQrte"]
