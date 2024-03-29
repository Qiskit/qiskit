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
            frame: Qiskit pulse Frame object to inquire.

        Returns:
            A unique frame identifier.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {frame}."
        )

    def get_port_identifier(
        self,
        logical_element: model.LogicalElement,
    ) -> str:
        """Return the unique port identifier of the Qiskit pulse logical element.

        Args:
            logical_element: Qiskit pulse LogicalElement object to inquire.

        Returns:
            A unique port identifier.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {logical_element}."
        )

    def is_mixed_frame_available(
        self,
        mixed_frame: model.MixedFrame,
    ) -> bool:
        """Check if given mixed frame is implementable on the hardware.

        Args:
            mixed_frame: Qiskit pulse MixedFrame object to test.

        Returns:
            True if given mixed frame is implementable.
        """
        return False

    def filter_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        logical_element: model.LogicalElement | None = None,
    ) -> list[model.MixedFrame]:
        """Filter available mixed frames on the hardware.

        Args:
            frame: Qiskit pulse Frame object to include.
            logical_element: Qiskit pulse LogicalElement object to include.

        Returns:
            A list of Qiskit pulse MixedFrame objects that include given frame and logical element.
        """
        return []

    def extra_frames(
        self,
        logical_element: model.LogicalElement,
    ) -> list[model.GenericFrame]:
        """Get a list of string identifier of unused frames 
        tied to the given Qiskit pulse logical element.

        Args:
            logical_element: Qiskit pulse LogicalElement object to inquire.

        Returns:
            A list of unique frame identifier.
        """
        return []


class QiskitPulseTarget(PulseTarget):
    """Qiskit reference implementation of :class:`.PulseTarget`."""

    def __init__(
        self,
        qubit_frames: dict[int, str] | None = None,
        meas_frames: dict[int, str] | None = None,
        qubit_ports: dict[int, str] | None = None,
        coupler_ports: dict[tuple[int, ...], str] | None = None,
        mixed_frames: dict[str, list[str]] | None = None,
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
            qubit_ports: A dictioanry of hardware port identifier keyed on
                the Qiskit :class:`~qiskit.pulse.model.Qubit` object index.
                This port must be used to drive qubit regardless of frames.
            coupler_ports: A dictionary of hardware port identifier keyed on
                the Qiskit :class:`.Port` object index.
                This port must be used to drive multi-qubit interactions.
            mixed_frams: A dictionary of avilable mixed frame resources keyed on
                the unique identifier of the port. Values are list of frame identifiers
                available for this port to form a mixed frame.

        """
        self._qubit_frames = qubit_frames
        self._qubit_frames_inv = dict(zip(qubit_frames.values(), qubit_frames.keys()))

        self._meas_frames = meas_frames
        self._meas_frames_inv = dict(zip(meas_frames.values(), meas_frames.keys()))

        self._qubit_ports = qubit_ports
        self._qubit_ports_inv = dict(zip(qubit_ports.values(), qubit_ports.keys()))

        self._coupler_ports = coupler_ports
        self._coupler_ports_inv = dict(zip(coupler_ports.values(), coupler_ports.keys()))

        self._mixed_frames = mixed_frames

    def is_mixed_frame_available(
        self,
        mixed_frame: model.MixedFrame,
    ) -> bool:
        if not isinstance(mixed_frame, model.MixedFrame):
            raise TypeError(f"{mixed_frame} is not a MixedFrame object.")
        try:
            p_uid = self.get_port_identifier(mixed_frame.pulse_target)
            f_uid = self.get_frame_identifier(mixed_frame.frame)
        except NotExistingComponent:
            return False
        try:
            return f_uid in self._mixed_frames[p_uid]
        except KeyError:
            return False

    def get_frame_identifier(
        self,
        frame: model.Frame,
    ) -> str:
        if isinstance(frame, model.QubitFrame):
            try:
                return self._qubit_frames[frame.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This hardware doesn't provide any frame for "
                    f"QubitFrame of index {frame.index}."
                ) from ex
        if isinstance(frame, model.MeasurementFrame):
            try:
                return self._meas_frames[frame.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This hardware doesn't provide any frame for "
                    f"MeasurementFrame of index {frame.index}."
                ) from ex
        raise TypeError(
            f"Input frame type {frame.__class__.__name__} cannot "
            "be directly mapped to hardware elements."
        )

    def get_port_identifier(
        self,
        logical_element: model.LogicalElement,
    ) -> str:
        if isinstance(logical_element, model.Qubit):
            try:
                return self._qubit_ports[logical_element.qubit_index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This hardware doesn't provide any port for "
                    f"Qubit of index {logical_element.qubit_index}."
                ) from ex
        if isinstance(logical_element, model.Coupler):
            try:
                return self._coupler_ports[logical_element.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This hardware doesn't provide any port for " 
                    f"Coupler of index {logical_element.index}."
                ) from ex
        raise TypeError(
            f"Input logical element type {logical_element.__class__.__name__} cannot "
            "be directly mapped to hardware elements."
        )

    def filter_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        logical_element: model.LogicalElement | None = None,
    ) -> list[model.MixedFrame]:
        try:
            if logical_element is not None:
                p_uid = self.get_port_identifier(logical_element)
            else:
                p_uid = None
            if frame is not None:
                f_uid = self.get_frame_identifier(frame)
            else:
                f_uid = None
        except NotExistingComponent:
            return []

        matched = []
        for port_name, frame_names in self._mixed_frames.items():
            if p_uid is not None and p_uid != port_name:
                continue
            for frame_name in frame_names:
                if f_uid is not None and f_uid != frame_name:
                    continue
                matched.append(
                    model.MixedFrame(
                        pulse_target=self._port_uid_to_obj(port_name),
                        frame=self._frame_uid_to_obj(frame_name),
                    )
                )
        return matched

    def extra_frames(
        self,
        logical_element: model.LogicalElement,
    ) -> list[model.GenericFrame]:
        p_uid = self.get_port_identifier(logical_element)
        try:
            frames = self._mixed_frames[p_uid]
        except KeyError as ex:
            raise NotExistingComponent(
                f"This hardware doesn't provide any mixed frame for the port {logical_element}."
            ) from ex
        out = []
        for frame in frames:
            if frame not in self._qubit_frames_inv and frame not in self._meas_frames_inv:
                out.append(model.GenericFrame(frame))
        return out

    def _port_uid_to_obj(self, port_uid: str) -> model.LogicalElement:
        if (index := self._qubit_ports_inv.get(port_uid, None)) is not None:
            return model.Qubit(index)
        if (index := self._coupler_ports.get(port_uid, None)) is not None:
            return model.Coupler(index)
        return model.Port(port_uid)

    def _frame_uid_to_obj(self, frame_uid: str) -> model.Frame:
        if (index := self._qubit_frames_inv.get(frame_uid, None)) is not None:
            return model.QubitFrame(index)
        if (index := self._meas_frames_inv.get(frame_uid, None)) is not None:
            return model.MeasurementFrame(index)
        return model.GenericFrame(frame_uid)
