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

"""
Delay instruction (for circuit module).
"""
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression


class Delay(Instruction):
    """Do nothing and just delay/wait/idle for a specified duration."""

    def __init__(self, duration, unit="dt"):
        """Create new delay instruction."""
        if unit not in {"s", "ms", "us", "ns", "ps", "dt"}:
            raise CircuitError("Unknown unit %s is specified." % unit)

        super().__init__("delay", 1, 0, params=[duration], unit=unit)

    def inverse(self):
        """Special case. Return self."""
        return self

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg for sublist in qargs for qarg in sublist], []

    def c_if(self, classical, val):
        raise CircuitError("Conditional Delay is not yet implemented.")

    @property
    def duration(self):
        """Get the duration of this delay."""
        return self.params[0]

    @duration.setter
    def duration(self, duration):
        """Set the duration of this delay."""
        self.params = [duration]

    def __array__(self, dtype=None):
        """Return the identity matrix."""
        return np.array([[1, 0], [0, 1]], dtype=dtype)

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the unitary matrix. This has been
        added to enable simulation without making delay a full Gate type.

        Returns:
            np.ndarray: matrix representation.
        """
        return self.__array__(dtype=complex)

    def __repr__(self):
        """Return the official string representing the delay."""
        return f"{self.__class__.__name__}(duration={self.params[0]}[unit={self.unit}])"

    def validate_parameter(self, parameter):
        """Delay parameter (i.e. duration) must be int, float or ParameterExpression."""
        if isinstance(parameter, int):
            if parameter < 0:
                raise CircuitError(
                    f"Duration for Delay instruction must be positive. Found {parameter}"
                )
            return parameter
        elif isinstance(parameter, float):
            if parameter < 0:
                raise CircuitError(
                    f"Duration for Delay instruction must be positive. Found {parameter}"
                )
            if self.unit == "dt":
                parameter_int = int(parameter)
                if parameter != parameter_int:
                    raise CircuitError("Integer duration is expected for 'dt' unit.")
                return parameter_int
            return parameter
        elif isinstance(parameter, ParameterExpression):
            if len(parameter.parameters) > 0:
                return parameter  # expression has free parameters, we cannot validate it
            if not parameter.is_real():
                raise CircuitError(f"Bound parameter expression is complex in delay {self.name}")
            fval = float(parameter)
            if fval < 0:
                raise CircuitError(f"Duration for Delay instruction must be positive. Found {fval}")
            if self.unit == "dt":
                ival = int(parameter)
                rounding_error = abs(fval - ival)
                if rounding_error > 1e-15:
                    raise CircuitError("Integer parameter is required for duration in 'dt' unit.")
                return ival
            return fval  # per default assume parameters must be real when bound
        else:
            raise CircuitError(f"Invalid param type {type(parameter)} for delay {self.name}.")
