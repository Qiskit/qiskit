# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
An AQC synthesis plugin to Qiskit's transpiler.
"""

from qiskit.transpiler.passes.synthesis import AQCSynthesisPlugin as NewAQCSynthesisPlugin
from qiskit.utils.deprecation import deprecate_func


class AQCSynthesisPlugin(NewAQCSynthesisPlugin):
    """
    An AQC-based Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"aqc"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    network_layout (str)
        Type of network geometry, one of {``"sequ"``, ``"spin"``, ``"cart"``, ``"cyclic_spin"``,
        ``"cyclic_line"``}. Default value is ``"spin"``.

    connectivity_type (str)
        type of inter-qubit connectivity, {``"full"``, ``"line"``, ``"star"``}.  Default value
        is ``"full"``.

    depth (int)
        depth of the CNOT-network, i.e. the number of layers, where each layer consists of a
        single CNOT-block.

    optimizer (:class:`~.Minimizer`)
        An implementation of the ``Minimizer`` protocol to be used in the optimization process.

    seed (int)
        A random seed.

    initial_point (:class:`~numpy.ndarray`)
        Initial values of angles/parameters to start the optimization process from.
    """

    @deprecate_func(
        since="0.46.0",
        pending=True,
        additional_msg="AQCSynthesisPlugin has been moved to qiskit.transpiler.passes.synthesis"
        "instead use AQCSynthesisPlugin from qiskit.transpiler.passes.synthesis",
    )
    def __init__(self):
        super().__init__()

    @deprecate_func(
        since="0.46.0",
        pending=True,
        additional_msg="AQCSynthesisPlugin has been moved to qiskit.transpiler.passes.synthesis"
        "instead use AQCSynthesisPlugin from qiskit.transpiler.passes.synthesis",
    )
    def run(self, unitary, **options):
        return super().run(unitary, **options)
