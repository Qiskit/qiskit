# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=================================================
Quantum programs (:mod:`qiskit.quantum_program`)
=================================================

.. currentmodule:: qiskit.quantum_program

A quantum program describes a hybrid quantum-classical computation as typed tensor values
produced and consumed by instructions. It is inert data: it describes a computation without
performing one.

A program's inputs and outputs travel in a data tree, arranged by the structures the program
declares, which is where all naming lives.

.. autoclass:: DataTree
"""

from qiskit._accelerate.quantum_program import DataTree

__all__ = ["DataTree"]
