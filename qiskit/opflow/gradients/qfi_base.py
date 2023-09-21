# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The module for Quantum the Fisher Information."""

from typing import Union

from qiskit.utils.deprecation import deprecate_func
from .derivative_base import DerivativeBase
from .circuit_qfis import CircuitQFI


class QFIBase(DerivativeBase):

    r"""Deprecated: Base class for Quantum Fisher Information (QFI).

    Compute the Quantum Fisher Information (QFI) given a pure, parameterized quantum state.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, qfi_method: Union[str, CircuitQFI] = "lin_comb_full"):
        r"""
        Args:
            qfi_method: The method used to compute the state/probability gradient. Can be either
                a :class:`CircuitQFI` instance or one of the following pre-defined strings
                ``'lin_comb_full'``, ``'overlap_diag'``` or ``'overlap_block_diag'```.
        Raises:
            ValueError: if ``qfi_method`` is neither a ``CircuitQFI`` object nor one of the
                predefined strings.
        """
        super().__init__()
        if isinstance(qfi_method, CircuitQFI):
            self._qfi_method = qfi_method

        elif qfi_method == "lin_comb_full":
            from .circuit_qfis import LinCombFull

            self._qfi_method = LinCombFull()
        elif qfi_method == "overlap_block_diag":
            from .circuit_qfis import OverlapBlockDiag

            self._qfi_method = OverlapBlockDiag()
        elif qfi_method == "overlap_diag":
            from .circuit_qfis import OverlapDiag

            self._qfi_method = OverlapDiag()
        else:
            raise ValueError(
                "Unrecognized input provided for `qfi_method`. Please provide"
                " a CircuitQFI object or one of the pre-defined string"
                " arguments: {'lin_comb_full', 'overlap_diag', "
                "'overlap_block_diag'}. "
            )

    @property
    def qfi_method(self) -> CircuitQFI:
        """Returns ``CircuitQFI``.

        Returns:
            ``CircuitQFI``.
        """
        return self._qfi_method
