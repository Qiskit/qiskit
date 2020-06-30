# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assemble function for converting a list of circuits into a qobj."""

from qiskit import qobj, pulse
from qiskit.pulse import lowering

from .run_config import RunConfig


def assemble_schedules(program: pulse.Program,
                       qobj_id: int,
                       qobj_header: qobj.QobjHeader,
                       run_config: RunConfig) -> qobj.PulseQobj:
    """Assembles a list of schedules into a qobj that can be run on the backend.

    Args:
        program: Pulse program.
        qobj_id: Identifier for the generated qobj.
        qobj_header: Header to pass to the results.
        run_config: Configuration of the runtime environment.

    Returns:
        The Qobj to be run on the backends.

    Raises:
        QiskitError: when frequency settings are not supplied.
    """
    return lowering.lower_qobj(
        program,
        qobj_id,
        qobj_header,
        run_config,
    )
