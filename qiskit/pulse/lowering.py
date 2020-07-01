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

"""Lowering compilation module for pulse programs."""
import copy
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from qiskit import pulse, qobj
from qiskit.assembler.run_config import RunConfig
from qiskit.pulse import (analysis, instructions, commands,
                          pulse_lib, transforms, validation)
from qiskit.pulse.basepasses import LoweringPass
from qiskit.pulse.compiler import BaseCompiler, Compiler, compile_result
from qiskit.pulse.exceptions import CompilerError
from qiskit.pulse.passmanager import PassManager
from qiskit.qobj import converters


def lower(
    program: pulse.Program,
    compiler: Optional[BaseCompiler] = None,
) -> Any:
    """Lower a pulse program."""
    return compile_result(program, compiler).lowered


def lower_qobj(
    program: pulse.Program,
    qobj_id: int,
    qobj_header: qobj.QobjHeader,
    run_config: RunConfig,
) -> qobj.PulseQobj:
    """Lower a pulse program."""
    compiler = QobjCompiler(
        qobj_id,
        qobj_header,
        run_config,
    )
    return lower(program, compiler)


class QobjCompiler(Compiler):
    """A qobj lowering compiler."""
    def __init__(
        self,
        qobj_id: int,
        qobj_header: qobj.QobjHeader,
        run_config: RunConfig,
    ):
        super().__init__()
        self.qobj_id = qobj_id
        self.qobj_header = qobj_header
        self.run_config = run_config

    def default_pipelines(self):
        super().default_pipelines()

        lowering_pm = PassManager()
        lowering_pm.append(
            LowerQobj(
                self.qobj_id,
                self.qobj_header,
                self.run_config,
            ),
        )

        self.append_pipeline(lowering_pm)


class LowerQobj(LoweringPass):
    """Return lowered qobj."""

    def __init__(
        self,
        qobj_id: int,
        qobj_header: qobj.QobjHeader,
        run_config: RunConfig,
    ):
        super().__init__()
        self.run_config = run_config
        self.qobj_id = qobj_id
        self.qobj_header = qobj_header

        self.requires.append(transforms.ConvertDeprecatedInstructions())
        self.requires.append(analysis.MaxMemorySlotUsed())
        self.requires.append(analysis.AmalgamatedAcquires())
        self.requires.append(
            transforms.NoInvalidParametricPulses(run_config.parametric_pulses),
        )
        if hasattr(run_config, 'meas_map'):
            self.requires.append(validation.ValidateMeasMap(run_config.meas_map))
        self.requires.append(transforms.DeDuplicateWaveformNames())

    def lower(
        self,
        program,
    ) -> qobj.PulseQobj:
        """Lower pulse program to Qobj.

        Returns:
            The Qobj to be run on the backends.

        Raises:
            CompilerError: when frequency settings are not supplied.
        """
        if not hasattr(self.run_config, 'qubit_lo_freq'):
            raise CompilerError('qubit_lo_freq must be supplied.')
        if not hasattr(self.run_config, 'meas_lo_freq'):
            raise CompilerError('meas_lo_freq must be supplied.')

        lo_converter = converters.LoConfigConverter(
            qobj.PulseQobjExperimentConfig,
            **self.run_config.to_dict(),
        )
        experiments, experiment_config = self._assemble_experiments(
            program,
            lo_converter,
        )
        qobj_config = self._assemble_config(
            lo_converter,
            experiment_config,
        )

        return qobj.PulseQobj(
            experiments=experiments,
            qobj_id=self.qobj_id,
            header=self.qobj_header,
            config=qobj_config,
        )

    def _assemble_experiments(
        self,
        program: pulse.Program,
        lo_converter: converters.LoConfigConverter,
    ) -> Tuple[List[qobj.PulseQobjExperiment], Dict[str, Any]]:
        """Assembles a list of schedules into PulseQobjExperiments, and returns
        related metadata that will be assembled into the Qobj configuration.

        Args:
            program: Program to compile.
            lo_converter: The configured frequency converter and validator.

        Returns:
            The list of assembled experiments, and the dictionary of related experiment config.

        Raises:
            QiskitError: when frequency settings are not compatible with the experiments.
        """
        freq_configs = [
            lo_converter(lo_dict) for
            lo_dict in getattr(self.run_config, 'schedule_los', [])]

        schedules = program.schedules

        if len(schedules) > 1 and len(freq_configs) not in [0, 1, len(schedules)]:
            raise CompilerError(
                'Invalid frequency setting is specified. If the frequency is specified, '
                'it should be configured the same for all schedules, configured for each '
                'schedule, or a list of frequencies should be provided for a single '
                'frequency sweep schedule.')

        instruction_converter = getattr(
            self.run_config,
            'instruction_converter',
            converters.InstructionToQobjConverter,
        )
        instruction_converter = instruction_converter(
            qobj.PulseQobjInstruction,
            **self.run_config.to_dict(),
        )

        user_pulselib = {}
        experiments = []
        for idx, schedule in enumerate(program.schedules):
            qobj_instructions = self._assemble_instructions(
                idx,
                schedule,
                instruction_converter,
                self.run_config,
                user_pulselib,
            )

            # TODO: add other experimental header items (see circuit assembler)
            qobj_experiment_header = qobj.QobjExperimentHeader(
                memory_slots=self.analysis.max_memory_slot_used[idx] + 1,  # Memory slots are 0 indexed
                name=schedule.name or 'Experiment-%d' % idx,
            )

            experiment = qobj.PulseQobjExperiment(
                header=qobj_experiment_header,
                instructions=qobj_instructions,
            )
            if freq_configs:
                # This handles the cases where one frequency setting applies to all experiments and
                # where each experiment has a different frequency
                freq_idx = idx if len(freq_configs) != 1 else 0
                experiment.config = freq_configs[freq_idx]

            experiments.append(experiment)

        # Frequency sweep
        if freq_configs and len(experiments) == 1:
            experiment = experiments[0]
            experiments = []
            for freq_config in freq_configs:
                experiments.append(
                    qobj.PulseQobjExperiment(
                        header=experiment.header,
                        instructions=experiment.instructions,
                        config=freq_config,
                        ),
                )

        # Top level Qobj configuration
        experiment_config = {
            'pulse_library': [
                qobj.PulseLibraryItem(name=name, samples=samples)
                for name, samples in user_pulselib.items()],
            'memory_slots': max([exp.header.memory_slots for exp in experiments])
        }

        return experiments, experiment_config

    def _assemble_instructions(
        self,
        idx: int,
        schedule: pulse.Schedule,
        instruction_converter: converters.InstructionToQobjConverter,
        run_config: RunConfig,
        user_pulselib: Dict[str, commands.Command]
    ) -> Tuple[List[qobj.PulseQobjInstruction], int]:
        """Assembles the instructions in a schedule into a list of PulseQobjInstructions and returns
        related metadata that will be assembled into the Qobj configuration. Lookup table for
        pulses defined in all experiments are registered in ``user_pulselib``. This object should be
        mutable python dictionary so that items are properly updated after each instruction assemble.
        The dictionary is not returned to avoid redundancy.

        Args:
            idx: Index of schedule in program.
            schedule: Schedule to assemble.
            instruction_converter: A converter instance which can convert PulseInstructions to
                                   PulseQobjInstructions.
            run_config: Configuration of the runtime environment.
            user_pulselib: User pulse library from previous schedule.

        Returns:
            A list of converted instructions, the user pulse library dictionary (from pulse name to
            pulse command), and the maximum number of readout memory slots used by this Schedule.
        """
        qobj_instructions = []

        for time, instruction in schedule.instructions:

            if isinstance(instruction, instructions.Play) and \
                    isinstance(instruction.pulse, pulse_lib.SamplePulse):
                name = hashlib.sha256(instruction.pulse.samples).hexdigest()
                instruction = instructions.Play(
                    pulse_lib.SamplePulse(
                        name=name,
                        samples=instruction.pulse.samples,
                    ),
                    channel=instruction.channel,
                    name=name,
                )
                user_pulselib[name] = instruction.pulse.samples

            if isinstance(instruction, (commands.AcquireInstruction, instructions.Acquire)):
                continue

            if isinstance(instruction, (commands.DelayInstruction, instructions.Delay)):
                # delay instructions are ignored as timing is explicit within qobj
                continue

            qobj_instructions.append(instruction_converter(time, instruction))

        acquire_instruction_map = self.analysis.acquire_instruction_maps[idx]
        if self.analysis.acquire_instruction_maps[idx]:
            for (time, _), instrs in acquire_instruction_map.items():
                qubits, mem_slots, reg_slots = self._bundle_channel_indices(instrs)
                qobj_instructions.append(
                    instruction_converter.convert_single_acquires(
                        time,
                        instrs[0],
                        qubits=qubits,
                        memory_slot=mem_slots,
                        register_slot=reg_slots,
                        ),
                    )

        return qobj_instructions

    def _bundle_channel_indices(
        self,
        instructions: List[commands.AcquireInstruction]
    ) -> Tuple[List[int], List[int], List[int]]:
        """From the list of AcquireInstructions, bundle the indices of the acquire channels,
        memory slots, and register slots into a 3-tuple of lists.

        Args:
            instructions: A list of AcquireInstructions to be bundled.

        Returns:
            The qubit indices, the memory slot indices, and register slot indices from instructions.
        """
        qubits = []
        mem_slots = []
        reg_slots = []
        for inst in instructions:
            qubits.append(inst.channel.index)
            if inst.mem_slot:
                mem_slots.append(inst.mem_slot.index)
            if inst.reg_slot:
                reg_slots.append(inst.reg_slot.index)
        return qubits, mem_slots, reg_slots

    def _assemble_config(
        self,
        lo_converter: converters.LoConfigConverter,
        experiment_config: Dict[str, Any],
    ) -> qobj.PulseQobjConfig:
        """Assembles the QobjConfiguration from experimental config and runtime config.

        Args:
            lo_converter: The configured frequency converter and validator.
            experiment_config: Schedules to assemble.
            run_config: Configuration of the runtime environment.

        Returns:
            The assembled PulseQobjConfig.
        """
        qobj_config = copy.copy(self.run_config.to_dict())
        qobj_config.update(experiment_config)

        # Run config not needed in qobj config
        qobj_config.pop('meas_map', None)
        qobj_config.pop('qubit_lo_range', None)
        qobj_config.pop('meas_lo_range', None)

        # convert enums to serialized values
        meas_return = qobj_config.get('meas_return', 'avg')
        if isinstance(meas_return, qobj.utils.MeasReturnType):
            qobj_config['meas_return'] = meas_return.value

        meas_level = qobj_config.get('meas_level', 2)
        if isinstance(meas_level, qobj.utils.MeasLevel):
            qobj_config['meas_level'] = meas_level.value

        # convert lo frequencies to Hz
        qobj_config['qubit_lo_freq'] = [freq / 1e9 for freq in qobj_config['qubit_lo_freq']]
        qobj_config['meas_lo_freq'] = [freq / 1e9 for freq in qobj_config['meas_lo_freq']]

        # frequency sweep config
        schedule_los = qobj_config.pop('schedule_los', [])
        if len(schedule_los) == 1:
            lo_dict = schedule_los[0]
            q_los = lo_converter.get_qubit_los(lo_dict)
            # Hz -> GHz
            if q_los:
                qobj_config['qubit_lo_freq'] = [freq / 1e9 for freq in q_los]
            m_los = lo_converter.get_meas_los(lo_dict)
            if m_los:
                qobj_config['meas_lo_freq'] = [freq / 1e9 for freq in m_los]

        return qobj.PulseQobjConfig(**qobj_config)
