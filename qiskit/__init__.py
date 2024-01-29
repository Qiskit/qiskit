# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=wrong-import-position

"""Main Qiskit public functionality."""

import sys
import warnings

import qiskit._accelerate


# Globally define compiled submodules. The normal import mechanism will not find compiled submodules
# in _accelerate because it relies on file paths, but PyO3 generates only one shared library file.
# We manually define them on import so people can directly import qiskit._accelerate.* submodules
# and not have to rely on attribute access.  No action needed for top-level extension packages.
sys.modules["qiskit._accelerate.nlayout"] = qiskit._accelerate.nlayout
sys.modules["qiskit._accelerate.quantum_circuit"] = qiskit._accelerate.quantum_circuit
sys.modules["qiskit._accelerate.stochastic_swap"] = qiskit._accelerate.stochastic_swap
sys.modules["qiskit._accelerate.sabre_swap"] = qiskit._accelerate.sabre_swap
sys.modules["qiskit._accelerate.sabre_layout"] = qiskit._accelerate.sabre_layout
sys.modules["qiskit._accelerate.pauli_expval"] = qiskit._accelerate.pauli_expval
sys.modules["qiskit._accelerate.dense_layout"] = qiskit._accelerate.dense_layout
sys.modules["qiskit._accelerate.sparse_pauli_op"] = qiskit._accelerate.sparse_pauli_op
sys.modules["qiskit._accelerate.results"] = qiskit._accelerate.results
sys.modules["qiskit._accelerate.optimize_1q_gates"] = qiskit._accelerate.optimize_1q_gates
sys.modules["qiskit._accelerate.sampled_exp_val"] = qiskit._accelerate.sampled_exp_val
sys.modules["qiskit._accelerate.vf2_layout"] = qiskit._accelerate.vf2_layout
sys.modules["qiskit._accelerate.error_map"] = qiskit._accelerate.error_map
sys.modules[
    "qiskit._accelerate.euler_one_qubit_decomposer"
] = qiskit._accelerate.euler_one_qubit_decomposer
sys.modules[
    "qiskit._accelerate.convert_2q_block_matrix"
] = qiskit._accelerate.convert_2q_block_matrix

# qiskit errors operator
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit import QuantumCircuit

# user config
from qiskit import user_config as _user_config

import qiskit.circuit.measure
import qiskit.circuit.reset

# Please note these are global instances, not modules.
from qiskit.providers.basicaer import BasicAer

_config = _user_config.get_config()

from qiskit.compiler import transpile, assemble, schedule, sequence

from .version import __version__

__all__ = [
    "AncillaRegister",
    "BasicAer",
    "ClassicalRegister",
    "MissingOptionalLibraryError",
    "QiskitError",
    "QuantumCircuit",
    "QuantumRegister",
    "assemble",
    "schedule",
    "sequence",
    "transpile",
]
