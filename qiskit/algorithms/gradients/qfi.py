# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
A class for the Quantum Fisher Information.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from copy import copy

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers import Options

from .base.base_qgt import BaseQGT
from .lin_comb.lin_comb_estimator_gradient import DerivativeType
from .qfi_result import QFIResult

from ..algorithm_job import AlgorithmJob
from ..exceptions import AlgorithmError


class QFI(ABC):
    r"""Computes the Quantum Fisher Information (QFI) given a pure,
    parameterized quantum state. QFI is defined as:

    .. math::

        \mathrm{QFI}_{ij}= 4 \mathrm{Re}[\langle \partial_i \psi | \partial_j \psi \rangle
        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].
    """

    def __init__(
        self,
        qgt: BaseQGT,
        options: Options | None = None,
    ):
        r"""
        Args:
            qgt: The quantum geometric tensor used to compute the QFI.
            options: Backend runtime options used for circuit execution. The order of priority is:
                options in ``run`` method > QFI's default options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        self._qgt: BaseQGT = qgt
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **options,
    ) -> AlgorithmJob:
        """Run the job of the QFIs on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the QFIs.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the QFIs of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the QFIs of all parameters in
                each circuit are calculated.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > QFI's
                default options > QGT's default setting.
                Higher priority setting overrides lower priority setting.

        Returns:
            The job object of the QFIs of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.
        """

        if isinstance(circuits, QuantumCircuit):
            # Allow a single circuit to be passed in.
            circuits = (circuits,)

        if parameters is None:
            # If parameters is None, we calculate the gradients of all parameters in each circuit.
            parameters = [circuit.parameters for circuit in circuits]
        else:
            # If parameters is not None, we calculate the gradients of the specified parameters.
            # None in parameters means that the gradients of all parameters in the corresponding
            # circuit are calculated.
            parameters = [
                params if params is not None else circuits[i].parameters
                for i, params in enumerate(parameters)
            ]
        # The priority of run option is as follows:
        # options in ``run`` method > QFI's default options > QGT's default setting.
        opts = copy(self._default_options)
        opts.update_options(**options)
        job = AlgorithmJob(self._run, circuits, parameter_values, parameters, **opts.__dict__)
        job.submit()
        return job

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> QFIResult:
        """Compute the QFI on the given circuits."""
        # Set the derivative type to real
        temp_derivative_type, self._qgt.derivative_type = (
            self._qgt.derivative_type,
            DerivativeType.REAL,
        )
        job = self._qgt.run(circuits, parameter_values, parameters, **options)

        try:
            result = job.result()
        except AlgorithmError as exc:
            raise AlgorithmError("Estimator job or gradient job failed.") from exc

        self._qgt.derivative_type = temp_derivative_type

        return QFIResult(
            qfis=[4 * qgt.real for qgt in result.qgts],
            metadata=result.metadata,
            options=result.options,
        )

    @property
    def options(self) -> Options:
        """Return the union of QGT's options setting and QFI's default options,
        where, if the same field is set in both, the QFI's default options override
        the QGT's default setting.

        Returns:
            The QFI default + QGT options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the gradient's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the QFI default setting,
        the QGT default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > QFI's default options > QGT's
        default setting.

        Args:
            options: The fields to update the options

        Returns:
            The QFI default + QGT default + run options.
        """
        opts = copy(self._qgt.options)
        opts.update_options(**options)
        return opts
