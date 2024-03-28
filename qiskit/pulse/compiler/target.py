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
    """Protocol of feeding hardware configuration to the Qiskit pulse compiler.

    This class defines interfaces for the Qiskit pulse compiler to receive
    necessary information to lower the input pulse programs.

    Because the system configuration is usually vendor dependent, we don't enforce
    the use of any Qiskit abstract class to give such vendors
    more flexibility to choose the most convenient data structure to
    represent their own device.

    The collection of interfaces is designed to provide all necessary information
    to convert the Qiskit pulse model into a vendor specific target bianry.

    Qiskit pulse uses an abstract model to represent hardware components, such as
    :class:`.QubitFrame` and :class:`qiskit.pulse.model.Qubit`.
    These components might be, not limited to, a software abstraction of
    a numerically controlled oscillator tracking the rotating frame of qubit,
    and a microwave or laser port connected with a particular qubit.

    This abstraction allows quantum physics experimentalists to use
    similar semantics with the ciruict quantum electrodynamics theory,
    while the compiler needs to lower those components down to the actual hardware resouces.

    We make a native assumption that such hardware devices provide unique string identifier
    for each component that it equips, and there are limited number of
    such physical resources. We also assume a frame is tie to ports on the hardware,
    namely, frames don't exist standalone and always form a mixed frame with a port in pair.

    See :class:`.QiskitPulseTarget` for the reference implementation.
    """

    def get_frame_identifier(
        self,
        frame: model.Frame,
    ) -> str:
        """Return the unique string identifier of the frame.

        Args:
            frame: Qiskit pulse Frame object to inquire.

        Returns:
            A unique identifier of given frame.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {frame}."
        )

    def get_port_identifier(
        self,
        port: model.LogicalElement,
    ) -> str:
        """Return the unique string identifier of the hardware port.

        Args:
            port: Qiskit pulse LogicalElement object to inquire.

        Returns:
            A unique identifier of given port.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {port}."
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
        port: model.LogicalElement | None = None,
    ) -> list[model.MixedFrame]:
        """Filter available mixed frames on the hardware.

        Args:
            frame: Qiskit pulse Frame object to include.
            port: Qiskit pulse LogicalElement object to include.

        Returns:
            A list of Qiskit pulse MixedFrame objects that include given frame and port.
        """
        return []

    def extra_frames(
        self,
        port: model.LogicalElement,
    ) -> list[model.GenericFrame]:
        """Get a list of string identifier of unused frames tied to the given port.

        Args:
            port: Qiskit pulse LogicalElement object to inquire.

        Returns:
            A list of unique identifier of unsed frames.
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
        port: model.LogicalElement,
    ) -> str:
        if isinstance(port, model.Qubit):
            try:
                return self._qubit_ports[port.qubit_index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This hardware doesn't provide any port for "
                    f"Qubit of index {port.qubit_index}."
                ) from ex
        if isinstance(port, model.Coupler):
            try:
                return self._coupler_ports[port.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This hardware doesn't provide any port for " f"Coupler of index {port.index}."
                ) from ex
        raise TypeError(
            f"Input port type {port.__class__.__name__} cannot "
            "be directly mapped to hardware elements."
        )

    def filter_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        port: model.LogicalElement | None = None,
    ) -> list[model.MixedFrame]:
        try:
            if port is not None:
                p_uid = self.get_port_identifier(port)
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
        port: model.LogicalElement,
    ) -> list[model.GenericFrame]:
        p_uid = self.get_port_identifier(port)
        try:
            frames = self._mixed_frames[p_uid]
        except KeyError as ex:
            raise NotExistingComponent(
                f"This hardware doesn't provide any mixed frame for the port {port}."
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
