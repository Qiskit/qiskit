# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for Pulse Compiler passes tests"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable

from qiskit.passmanager import BasePassManager
from qiskit.pulse.ir import SequenceIR


class PulseIrTranspiler(BasePassManager, ABC):
    """Pass manager for Pulse IR -> Pulse IR transpilation"""

    def _passmanager_frontend(
        self,
        input_program: SequenceIR,
        **kwargs,
    ) -> SequenceIR:

        return input_program

    def _passmanager_backend(
        self,
        passmanager_ir: SequenceIR,
        in_program: SequenceIR,
        **kwargs,
    ) -> SequenceIR:

        return passmanager_ir

    # pylint: disable=arguments-differ
    def run(
        self,
        pulse_programs: SequenceIR | list[SequenceIR],
        callback: Callable | None = None,
        num_processes: int | None = None,
    ) -> SequenceIR | list[SequenceIR]:
        """Run all the passes on the input pulse programs."""
        return super().run(
            in_programs=pulse_programs,
            callback=callback,
            num_processes=num_processes,
        )
