from collections import defaultdict
from typing import Tuple, Iterable, Union

from qiskit import schedule, pulse
from qiskit.circuit import Gate
from qiskit.pulse import Play, Acquire
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule




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
        # TODO: fix below

        # schedule.draw()
        if isinstance(instruction, Gate):

            if self._map[instruction.name]:
                # self.assert_has(instruction.name, qubits)
                # TODO: copied to_tuple because protected but feels redundent?
                schedule_generator = self._map[instruction.name].get(_to_tuple(qubits))
            else:

                print('gate hit')
                pulse_seq = (self.qoc_optimizer.get_pulse_schedule(instruction))
                out_schedule = pulse.Schedule()
                drive_chan = pulse.DriveChannel(1)
                # Figure out universal version for more drive channels

                out_schedule += Play(pulse.SamplePulse(pulse_seq), drive_chan) << out_schedule.duration
                schedule_generator = out_schedule
        else:
            self.assert_has(instruction, qubits)
            # TODO: copied to_tuple because protected but feels redundent?
            schedule_generator = self._map[instruction].get(_to_tuple(qubits))
        # don't forget in here to use _gate.to_matrix

        if callable(schedule_generator):
            return schedule_generator(*params, **kwparams)
        # otherwise this is just a Schedule
        return schedule_generator

    @classmethod
    def from_inst_map(cls, grape_optimizer, instruction_schedule_map, default_inst=['measure']):
        # probably replace this using get for measurement
        imap = {gate: instruction_schedule_map._map[gate] for gate in default_inst}
        imap = defaultdict(str, imap)

        qubit_instructions = instruction_schedule_map._qubit_instructions

        qoc_map = QOCInstructionScheduleMap(grape_optimizer)
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
