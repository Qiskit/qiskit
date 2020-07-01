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
from abc import abstractmethod
from typing import Optional

from qiskit import pulse
from qiskit.pulse import transforms
from qiskit.pulse.passmanager import PassManager
from qiskit.pulse.states import AttrDict, State


class CompilerResult(AttrDict):
    """Result from the compiler."""
    def __init__(self, pulse_program, analysis, lowered=None):
        self.pulse_program = pulse_program
        self.analysis = analysis
        self.lowered = lowered


class BaseCompiler:
    """The default pulse compiler."""

    def __init__(self):
        self.state = State()
        self._pipelines = []
        self._finalized = False

    @property
    def pipelines(self):
        """Return compiler pipelines"""
        return self._pipelines

    def append_pipeline(self, pipeline: PassManager):
        """Add compiler pipeline."""
        self._pipelines.append(pipeline)

    def compile(self, program: pulse.Program) -> pulse.Program:
        self.finalize()
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

    @abstractmethod
    def default_pipelines(self):
        """Build the default pipelines for the compiler."""

    def finalize(self):
        """Finalize the compiler before compilation if not already done."""
        if not self._finalized:
            self.default_pipelines()


class Compiler(BaseCompiler):
    """The default pulse compiler."""
    def default_pipelines(self):
        """Build the default pipelines."""
        self.append_pipeline(PipeLineBuilder.default_optimization_pipeline())


class PipeLineBuilder:
    """Builder for standard pipelines."""
    @staticmethod
    def default_optimization_pipeline() -> PassManager:
        pm = PassManager()
        pm.append(transforms.ConvertDeprecatedInstructions())
        pm.append(transforms.CompressPulses())
        return pm


def compile_result(
    program: pulse.Program,
    compiler: BaseCompiler,
) -> CompilerResult:
    """Compile a pulse program returning the compiler result."""
    return compiler.compile(program)


def compile(
    program: pulse.Program,
    compiler: Optional[Compiler] = None,
) -> pulse.Program:
    """Compile a pulse program."""
    compiler = compiler or Compiler()
    return compile_result(program, compiler).pulse_program
