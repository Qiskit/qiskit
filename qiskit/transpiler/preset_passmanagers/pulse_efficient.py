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

"""Pass manager for pulse-efficient, cross-resonance gate based circuits.

Pulse-efficient pass manager: Computes Weyl parameters of two-qubit operations in a circuit
and translates the circuit into a pulse-efficient, cross-resonance gate based circuit by creating
calibrations for the cross-resonance gates. Reference: https://arxiv.org/pdf/2105.01063.pdf
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks

from qiskit.transpiler.passes.scheduling.calibration_creators import RZXCalibrationBuilderNoEcho
from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import EchoRZXWeylDecomposition
from qiskit.circuit.library.standard_gates.equivalence_library import StandardEquivalenceLibrary as std_eqlib
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions
from qiskit.transpiler.passes import Optimize1qGatesDecomposition


def pulse_efficient_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Pulse-efficient pass manager.

    This pass manager consolidates all consecutive two-qubit operations in a quantum circuit and
    computes the Weyl decomposition of the corresponding unitaries. It rewrites the unitary operations
    in terms of cross-resonance gates and creates calibrations for these gates. Lastly, the circuit
    is rewritten in the hardware-native basis ['rzx', 'rz', 'x', 'sx'].

    Note:
        This pass manager does currently not support the transpilation of two-qubit operations
        between qubits that are not connected on the hardware.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a pulse-efficient pass manager.
    """
    backend = pass_manager_config.backend

    # 1. Consolidate all consecutive two-qubit operations
    _collect_2q_blocks = Collect2qBlocks()
    _consolidate_blocks = ConsolidateBlocks(basis_gates=['rz', 'sx', 'x', 'rxx'])

    # 2. Decompose two-qubit unitaries in terms of echoed RZX gates according to the Weyl decomposition
    _echo_rzx_weyl_decomposition = EchoRZXWeylDecomposition(backend)

    # 3. Add calibrations
    _rzx_calibrations = RZXCalibrationBuilderNoEcho(backend)

    # 4. Unroll to ['rzx', 'rz', 'x', 'sx'] basis
    rzx_basis = ['rzx', 'rz', 'x', 'sx']
    _unroll = [UnrollCustomDefinitions(std_eqlib, rzx_basis),
               BasisTranslator(std_eqlib, rzx_basis)]

    # 5. Optimize one-qubit decomposition
    _optimize_1q_decomposition = Optimize1qGatesDecomposition(rzx_basis)

    # Build pass manager
    pm0 = PassManager()
    pm0.append(_collect_2q_blocks)
    pm0.append(_consolidate_blocks)
    pm0.append(_echo_rzx_weyl_decomposition)
    pm0.append(_rzx_calibrations)
    pm0.append(_unroll)
    pm0.append(_optimize_1q_decomposition)
    return pm0
