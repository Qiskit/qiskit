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

"""Compiler target information."""

from __future__ import annotations

from typing import Protocol
from qiskit.pulse import model
from qiskit.pulse.exceptions import NotExistingComponent


class PulseTarget(Protocol):
    """Protocol of feeding the Qiskit pulse compiler with hardware configurations.

    This class defines interfaces for the Qiskit pulse compiler to receive
    necessary information to lower the input pulse programs.

    Because the system configuration is usually vendor dependent, we don't enforce
    the use of any Qiskit abstract class to give such vendors
    more flexibility to choose the most convenient data structure to
    represent their own devices.

    The collection of interfaces is designed to provide all necessary information
    to gurantee the builtin Qiskit pulse compiler passes can convert
    the Qiskit pulse program representation into a vendor specific target binary.
    Qiskit pulse uses an abstract model to represent hardware components, such as
    :class:`.QubitFrame` and :class:`qiskit.pulse.model.Qubit`.
    These components might be, not limited to, a software abstraction of
    a numerically controlled oscillator tracking the rotating frame of qubit,
    and a microwave or laser element connected with a particular qubit.

    This abstraction allows quantum physics experimentalists to adopt
    similar semantics to the familiar ciruict quantum electrodynamics theory
    without concerning about the detailed setup of the hardware device,
    while the Qiskit compiler needs to lower those components down to the actual equipment.

    Here we make native assumptions. A provider of this taregt must design
    its data structure to meet following rules otherwise the compiled program may cause
    faulty behavior. The hardware devices, or pulse controller, 
    must distinghish every hardware ports and frames with unique string identifiers, and 
    frames are tied to ports. In other words, frames don't exist standalone and 
    always can form a mixed frame with a port in pair.
    
    In addition, Qiskit provides special preset frame types, 
    :class:`.QubitFrame` and :class:`.MeasurementFrame`, and the backend must provide 
    special string identifiers for these frame objects. 
    If there is any architecture specific frames, a vendor can also define custom 
    frame subclasses to extend the target model to align with their hardware devices.
    If the backend defines any calibration for its basis gates, 
    we assume the same string identifier is used within the calibration so that
    a user provided inline calibration (pulse gates) can work seamlessly 
    with the backend calibrated gate sets.

    See :class:`.QiskitPulseTarget` for the reference implementation.
    """

    def get_frame_identifier(
        self,
        frame: model.Frame,
    ) -> str:
        """Return the unique frame identifier of the Qiskit pulse frame.

        Args:
            frame: Qiskit pulse frame to inquire.

        Returns:
            A unique frame identifier.

        Raises:
            NotExistingComponent: When the frame identifier is not found.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {frame}."
        )
    
    def get_generic_port_identifier(
        self,
        signal_entry: model.SignalEntry,        
    ) -> str:
        """Return the unique port identifier of the Qiskit pulse signal entry
        for generic operations.

        Args:
            signal_entry: Qiskit pulse mixed frame to inquire.

        Returns:
            A unique port identifier.

        Raises:
            NotExistingComponent: When the frame identifier is not found.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {signal_entry}."
        )

    def get_measure_port_identifier(
        self,
        signal_entry: model.SignalEntry,        
    ) -> str:
        """Return the unique port identifier of the Qiskit pulse signal entry
        for measurement operations.

        Args:
            signal_entry: Qiskit pulse mixed frame to inquire.

        Returns:
            A unique port identifier.

        Raises:
            NotExistingComponent: When the frame identifier is not found.
        """  
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {signal_entry}."
        )

    def reserved_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        signal_entry: model.SignalEntry | None = None,        
    ) -> list[model.MixedFrame]:
        """Return a list of mixed frames reserved for the backend gate calibrations.
        
        Args:
            frame: Qiskit pulse Frame object to include.
            signal_entry: Qiskit pulse SignalEntry object to include.

        Returns:
            A list of Qiskit pulse mixed frame objects that 
            include given frame and logical element.
        """
        return []


class QiskitPulseTarget(PulseTarget):
    """Qiskit reference implementation of :class:`.PulseTarget`."""

    def __init__(
        self,
        qubit_frames: dict[int, str] | None = None,
        meas_frames: dict[int, str] | None = None,
        tx_ports: dict[str, dict] | None = None,
        rx_ports: dict[str, dict] | None = None,
    ) -> None:
        """Create new Qiskit pulse target.

        Args:
            qubit_frames: A dictionary of qubit frame identifier keyed on
                the Qiskit :class:`.QubitFrame` object index.
                This frame must track the rotating frame
                of the qubit control signal.
            meas_frames: A dictionary of measurement frame identifier keyed on
                the Qiskit :class:`.MeasurementFrame` object index.
                This frame must track the rotating frame
                of the measurement stimulus signal.
            tx_ports: A list of dictionary representing a spec of transmission ports. 
                The spec dictionary must contain "qubits", "op_type", "num_frames",
                "reserved_frames" keyes.
            rx_ports: A list of dictionary representing a spec of receiver ports.
                The spec dictionary must contain ...
        """
        self._qubit_frames = qubit_frames
        self._qubit_frames_inv = dict(zip(qubit_frames.values(), qubit_frames.keys()))

        self._meas_frames = meas_frames
        self._meas_frames_inv = dict(zip(meas_frames.values(), meas_frames.keys()))

        # Use table-like data format, if we can add to dependency.
        # In realistic provider implementation use of schema is recommended.
        self._tx_ports = tx_ports

    def get_frame_identifier(
        self,
        frame: model.Frame,
    ) -> str:
        if isinstance(frame, model.GenericFrame):
            return frame.name
        if isinstance(frame, model.QubitFrame):
            try:
                return self._qubit_frames[frame.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This control system doesn't provide any frame for "
                    f"implementing QubitFrame of index {frame.index}."
                ) from ex
        if isinstance(frame, model.MeasurementFrame):
            try:
                return self._meas_frames[frame.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This control system doesn't provide any frame for "
                    f"implementing MeasurementFrame of index {frame.index}."
                ) from ex
        raise TypeError(
            f"{self.__class__.__name__} doesn't recognize the frame object of type "
            f"{frame.__class__.__name__}. If you are using a custom subclass, "
            "you must also define PulseTarget class supporting this type."
        )

    def get_generic_port_identifier(
        self,
        signal_entry: model.SignalEntry,        
    ) -> str:
        return self._get_port_common(signal_entry, "generic")

    def get_measure_port_identifier(
        self,
        signal_entry: model.SignalEntry,        
    ) -> str:
        return self._get_port_common(signal_entry, "measure")

    def _get_port_common(
        self,
        signal_entry: model.SignalEntry,
        op_type: str,
    ) -> str:
        if isinstance(signal_entry, model.Port):
            if signal_entry.name not in self._tx_ports:
                raise NotExistingComponent(
                    f"Port identifier {signal_entry.name} is not defined in this system. "
                    "Hardware may not implement this port."
                )
            return signal_entry.name        
        if isinstance(signal_entry, model.LogicalElement):
            qubit_index = list(signal_entry.index)
            for port_uid, data in self._tx_ports.items():
                if data["qubits"] == qubit_index and data["op_type"] == op_type:
                    return port_uid
            else:
                raise NotExistingComponent(
                    "This control system doesn't provide any port for "
                    f"implementing LogicalElement {signal_entry}."
                )
        raise TypeError(
            f"{self.__class__.__name__} doesn't recognize the signal entry object of type "
            f"{signal_entry.__class__.__name__}. If you are using a custom subclass, "
            "you must also define PulseTarget class supporting this type."
        )        

    def reserved_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        signal_entry: model.SignalEntry | None = None,        
    ) -> list[model.MixedFrame]:
        # TODO implement this
