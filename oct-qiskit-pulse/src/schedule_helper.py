from qiskit import pulse
from qiskit.pulse import Play, Schedule

from typing import Union, List, Callable, Mapping

def channel_finder(channel_string: str) -> Union[pulse.DriveChannel, pulse.ControlChannel]:
    """Convert a channel string to a real channel

    Raises: NotImplementedError: If there is some other channel like a
        measurement channel, which we have not implemented, we raise
        NotImplementedError.

    Returns: (Union[pulse.DriveChannel, pulse.ControlChannel]): The resulting channel.
    """
    channel_type = channel_string[0]
    qubit = int(channel_string[1:])
    if channel_type == 'D':
        channel = pulse.DriveChannel(qubit)
    elif channel_type == 'U':
        channel = pulse.ControlChannel(qubit)
    else:
        raise NotImplementedError("Unknown channel type encountered.")
    return channel


def sequence_converter(pulse_seq_dict: Mapping[str, List[complex]]) -> Schedule:
    """Convert a dictionary of pulses and channels to a pulse schedule.

    Args:
        pulse_seq_dict (Mapping[str, List[complex): A dictionary of pulses and corresponding channels.

    Returns:
        Schedule: A pulse schedule.
    """
    out_schedule = Schedule()
    for channel in pulse_seq_dict.keys():
        channel_obj = channel_finder(channel)
        out_schedule += Play(pulse.SamplePulse(pulse_seq_dict[channel]), channel_obj)
    return out_schedule


def qutip_amps_to_channels(result_amps):
    channels = result_amps.keys()
    output={}
    for channel in channels:
        if 'y' not in channel:
            ychannel = channel + 'y'
            if ychannel in channels:
                output[channel] = [complex(a[0],a[1]) for a in zip(result_amps[channel], result_amps[ychannel])]
            else:
                output[channel] = result_amps
    return output
