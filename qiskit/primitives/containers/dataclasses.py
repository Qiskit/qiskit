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
"""
Dataclass
"""

from qiskit.utils.optionals import HAS_PYDANTIC

if HAS_PYDANTIC:
    from pydantic import ConfigDict
    from pydantic.dataclasses import dataclass

    mutable_dataclass = dataclass(
        config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, extra="forbid")
    )

    frozen_dataclass = dataclass(
        config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"),
        frozen=True,
        slots=True,
    )
else:
    from dataclasses import dataclass

    mutable_dataclass = dataclass(frozen=False)
    frozen_dataclass = dataclass(frozen=True)
