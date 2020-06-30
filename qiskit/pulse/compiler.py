# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compiler module for pulse schedules."""

from qiskit import pulse
from qiskit.pulse.passmanager import PassManager
from qiskit.pulse.states import AttrDict, State


class CompilerResult(AttrDict):
    """Result from the compiler."""
    def __init__(self, pulse_program, analysis, lowered=None):
        self.pulse_program = pulse_program
        self.analysis = analysis
        self.lowered = lowered


class Compiler:
    """The default pulse compiler."""

    def __init__(self):
        self.state = State()
        self._pipelines = []

    @property
    def pipelines(self):
        """Return compiler pipelines"""
        return self._pipelines

    def append_pipeline(self, pipeline: PassManager):
        """Add compiler pipeline."""
        self._pipelines.append(pipeline)

    def compile(self, program: pulse.Program) -> pulse.Program:
        self.state.program = program
        for pipeline in self.pipelines:
            program = pipeline.run(program)
            self.state = pipeline.state
            self.state.program = program

        return CompilerResult(
            program,
            self.state.analysis,
            lowered=self.state.lowered,
        )

    def default_pipelines(self):
        """Build the default pipelines."""
        self.pipelines.append_pipeline(PipeLineBuilder.default_optimization_pipeline())


class PipeLineBuilder:
    """Builder for standard pipelines."""
    @staticmethod
    def default_optimization_pipeline() -> PassManager:
        return PassManager()
