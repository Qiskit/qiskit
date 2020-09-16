# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class implementing a (piecewise-) linear function on qubit amplitudes."""

from typing import Optional, List, Union, Tuple
import numpy as np
from qiskit.circuit import QuantumCircuit

from .piecewise_linear_pauli_rotations import PiecewiseLinearPauliRotations


class LinearAmplitudeFunction(QuantumCircuit):
    """A circuit implementing a (pieceswie) linear function on qubit amplitudes.

    This class uses Pauli-Y rotations to rotate the values onto the amplitudes.
    """

    def __init__(self,
                 num_state_qubits: int,
                 slope: Union[float, List[float]],
                 offset: Union[float, List[float]],
                 domain: Tuple[float, float],
                 image: Tuple[float, float],
                 rescaling_factor: float = 1,
                 breakpoints: Optional[List[float]] = None,
                 name: str = 'F',
                 ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits used to encode the variable :math:`x`.
            slope: The slope of the linear function. Can be a list of slopes if it is a piecewise
                linear function.
            offset: The offset of the linear function. Can be a list of offsets if it is a piecewise
                linear function.
            domain: The domain of the function as tuple :math:`(x_\min{}, x_\max{})`.
            image: The image of the function as tuple :math:`(f_\min{}, f_\max{})`.
            rescaling_factor: The rescaling factor :math:`c`.
            breakpoints: The breakpoints if the function is piecewise linear. If None, the function
                is not piecewise.
            name: Name of the circuit.
        """
        if isinstance(slope, float):
            slope = [slope]
        if isinstance(offset, float):
            offset = [offset]

        _check_sizes_match(slope, offset, breakpoints)
        _check_sorted_and_in_range(breakpoints, domain)

        # ensure that the breakpoints include the first point of the domain
        if breakpoints is None:
            breakpoints = [domain[0]]
        else:
            if not np.isclose(breakpoints[0], domain[0]):
                breakpoints = [domain[0]] + breakpoints
                slopes = [0] + slopes
                offsets = [0] + offsets

        # do rescalings
        a, b = domain
        c, d = image

        mapped_breakpoints = []
        mapped_slopes = []
        mapped_offsets = []
        for i, point in enumerate(breakpoints):
            mapped_breakpoint = (point - a) / (b - a) * (2**num_state_qubits - 1)
            if mapped_breakpoint <= 2**num_state_qubits - 1:
                mapped_breakpoints += [mapped_breakpoint]

                # factor (upper - lower) / (2^n - 1) is for the scaling of x to [l,u]
                # note that the +l for mapping to [l,u] is already included in
                # the offsets given as parameters
                mapped_slopes += [slopes[i] * (b - a) / (2**num_state_qubits - 1)]
                mapped_offsets += [offsets[i]]
            else:
                raise RuntimeError('wtf')

        # approximate linear behavior by scaling and contracting around pi/4
        slope_angles = np.zeros(len(breakpoints))
        offset_angles = np.pi / 4 * (1 - rescaling_factor) * np.ones(len(breakpoints))
        for i in range(len(breakpoints)):
            slope_angles[i] = np.pi * rescaling_factor * mapped_slopes[i] / 2 / (d - c)
            offset_angles[i] += np.pi * rescaling_factor * (mapped_offsets[i] - c) / 2 / (d - c)

        # use PWLPauliRotations to implement the function
        pwl_pauli_rotation = PiecewiseLinearPauliRotations(
            num_state_qubits,
            mapped_breakpoints,
            2 * slope_angles,
            2 * offset_angles
        )

        super().__init__(pwl_pauli_rotation.num_qubits, name=name)
        self.compose(pwl_pauli_rotation, inplace=True)


def _check_sorted_and_in_range(breakpoints, domain):
    # check if sorted
    if not np.all(np.diff(breakpoints) > 0):
        raise ValueError('Breakpoints must be unique and sorted.')

    if breakpoints[0] < domain[0] or breakpoints[-1] > domain[1]:
        raise ValueError('Breakpoints must be included in domain.')


def _check_sizes_match(slope, offset, breakpoints):
    size = len(slope)
    if len(offset) != size:
        raise ValueError('Size mismatch of slope ({}) and offset ({}).'.format(size, len(offset)))
    if breakpoints is not None:
        if len(slope) != size:
            raise ValueError('Size mismatch of slope ({}) and breakpoints ({}).'.format(
                size, len(breakpoints)))
