# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Add single-pulse RX calibrations that are bootstrapped from the SX calibration."""

from typing import Union
from functools import lru_cache
import numpy as np

from qiskit.circuit import Instruction
from qiskit.pulse import Schedule, ScheduleBlock, builder
from qiskit.pulse.channels import Channel
from qiskit.pulse.library.symbolic_pulses import Drag
from qiskit.transpiler.passes.calibration.base_builder import CalibrationBuilder
from qiskit.transpiler import Target
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.exceptions import QiskitError


class RXCalibrationBuilder(CalibrationBuilder):
    """Add single-pulse RX calibrations that are bootstrapped from the SX calibration.

    .. note::

        Requirement: NormalizeRXAngles pass (one of the optimization passes).

    It is recommended to place this pass in the post-optimization stage of a passmanager.
    A simple demo:

    .. code-block:: python

       backend = FakeBelemV2()
       pm = PassManager(RXCalibrationBuilder(backend.target))
       qc = QuantumCircuit(1)
       angles = [0.1, 0.2, 0.3, 0.4]
       for angle in angles:
         qc.rx(angle, 0)

       # run the pass and check that new calibrations are generated
       transpiled_circuit = pm.run(qc)
       print(transpiled_circuit.calibrations["rx"])

    References
        * [1]: Gokhale et al. (2020), Optimized Quantum Compilation for
          Near-Term Algorithms with OpenPulse.
          `arXiv:2004.11205 <https://arxiv.org/abs/2004.11205>`
    """

    def __init__(
        self,
        target: Target = None,
    ):
        """Bootstrap single-pulse RX gate calibrations from the
        (hardware-calibrated) SX gate calibration.

        Args:
            target (Target): Should contain a SX calibration that will be
            used for bootstrapping RX calibrations.
        """
        from qiskit.transpiler.passes.optimization import NormalizeRXAngle

        super().__init__()
        self.target = target
        self.already_generated = {}
        self.requires = [NormalizeRXAngle(self.target)]

    def supported(self, node_op: Instruction, qubits: list) -> bool:
        """
        Check if the calibration for SX gate exists.
        """
        return isinstance(node_op, RXGate) and self.target.has_calibration("sx", tuple(qubits))

    def get_calibration(self, node_op: Instruction, qubits: list) -> Union[Schedule, ScheduleBlock]:
        """
        Generate RX calibration for the rotation angle specified in node_op.
        """
        # already within [0, pi] by NormalizeRXAngles pass
        angle = node_op.params[0]

        try:
            angle = float(angle)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        params = (
            self.target.get_calibration("sx", tuple(qubits))
            .instructions[0][1]
            .pulse.parameters.copy()
        )
        new_rx_sched = _create_rx_sched(
            rx_angle=angle,
            channel=self.target.get_calibration("sx", tuple(qubits)).channels[0],
            duration=params["duration"],
            amp=params["amp"],
            sigma=params["sigma"],
            beta=params["beta"],
        )

        return new_rx_sched


@lru_cache
def _create_rx_sched(
    rx_angle: float,
    duration: int,
    amp: float,
    sigma: float,
    beta: float,
    channel: Channel,
):
    """Generates (and caches) pulse calibrations for RX gates.
    Assumes that the rotation angle is in [0, pi].
    """
    new_amp = rx_angle / (np.pi / 2) * amp
    with builder.build() as new_rx_sched:
        builder.play(
            Drag(duration=duration, amp=new_amp, sigma=sigma, beta=beta, angle=0),
            channel=channel,
        )

    return new_rx_sched
