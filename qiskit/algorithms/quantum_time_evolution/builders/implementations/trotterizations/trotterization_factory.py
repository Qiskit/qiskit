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
"""TrotterizationFactory Class """
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.qdrift \
    import \
    QDrift
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.suzuki \
    import \
    Suzuki
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.trotter \
    import \
    Trotter
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations\
    .trotter_mode_enum import \
    TrotterModeEnum
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations\
    .trotterization_base import \
    TrotterizationBase


class TrotterizationFactory:
    """A factory for conveniently creating TrotterizationBase instances."""

    @staticmethod
    def build(mode: TrotterModeEnum = TrotterModeEnum.TROTTER, reps: int = 1) -> TrotterizationBase:
        """A factory for conveniently creating TrotterizationBase instances.
        Args:
            mode: One of 'trotter', 'suzuki', 'qdrift'
            reps: The number of times to repeat the Trotterization circuit.
        Returns:
            The desired TrotterizationBase instance.
        Raises:
            ValueError: A string not in ['trotter', 'suzuki', 'qdrift'] is given for mode.
        """
        if mode == TrotterModeEnum.TROTTER:
            return Trotter(reps=reps)

        elif mode == TrotterModeEnum.SUZUKI:
            return Suzuki(reps=reps)

        elif mode == TrotterModeEnum.QDRIFT:
            return QDrift(reps=reps)

        raise ValueError(f"Trotter mode {mode.value} not supported")