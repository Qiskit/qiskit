from collections import defaultdict
from typing import Tuple, Iterable, Union

from qiskit import schedule, pulse
from qiskit.circuit import Gate
from qiskit.pulse import Play, Acquire
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule

from src.schedule_helper import sequence_converter


class QOCInstructionScheduleMap(InstructionScheduleMap):
    def __init__(self, qoc_optimizer):
        super().__init__()
        self.qoc_optimizer = qoc_optimizer
        # FIXME don't use protected member
        # self._new_map =

    def get(self,
            # TODO figure out type hints for gate below
            instruction: Union[str, Gate],
            qubits: Union[int, Iterable[int]],
            *params: Union[int, float, complex],
            **kwparams: Union[int, float, complex]) -> Schedule:
        """Return the defined :py:class:`~qiskit.pulse.Schedule` for the given instruction on
        the given qubits.

        Args:
            instruction: Name of the instruction.
            qubits: The qubits for the instruction.
            *params: Command parameters for generating the output schedule.
            **kwparams: Keyworded command parameters for generating the schedule.

        Returns:
            The Schedule defined for the input.
        """
        if isinstance(instruction, Gate):
            if self._map[instruction.name]:
                # self.assert_has(instruction.name, qubits)
                schedule_generator = self._map[instruction.name].get(_to_tuple(qubits))
            else:

                pulse_seq_with_channels = (self.qoc_optimizer.get_pulse_schedule(instruction, qubits))
                schedule_generator = sequence_converter(pulse_seq_with_channels)
        else:
            self.assert_has(instruction, qubits)
            schedule_generator = self._map[instruction].get(_to_tuple(qubits))

        if callable(schedule_generator):
            return schedule_generator(*params, **kwparams)
        return schedule_generator

    @classmethod
    def from_inst_map(cls, optimizer, instruction_schedule_map, default_inst=['measure']):
        """Instantiate a QOCInstructionScheduleMap with some builtin instructions
        Args:
            optimizer: QOCOptimizer
                The optimizer to use for this instruction mapping
            instruction_schedule_map: InstructionScheduleMap
                The InstructionScheduleMap from which we pull the default pulse 
                instructions. Usually used for measurement.
            default_inst: Array[str]
                A list of instruction mappings to take from the default instruction mapping. 
                Usually just ['measure']
        Returns:
            The created QOCInstructionScheduleMap
        """
        # probably replace this using get for measurement
        imap = {gate: instruction_schedule_map._map[gate] for gate in default_inst}
        imap = defaultdict(str, imap)

        qubit_instructions = instruction_schedule_map._qubit_instructions

        qoc_map = QOCInstructionScheduleMap(optimizer)
        qoc_map._map = imap
        qoc_map._qubit_instructions = qubit_instructions
        return qoc_map


def _to_tuple(values: Union[int, Iterable[int]]) -> Tuple[int, ...]:
    """Return the input as a tuple.

    Args:
        values: An integer, or iterable of integers.

    Returns:
        The input values as a sorted tuple.
    """
    try:
        return tuple(values)
    except TypeError:
        return (values,)
