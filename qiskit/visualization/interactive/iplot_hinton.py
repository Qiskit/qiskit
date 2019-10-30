# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Hinton visualization
"""
from string import Template
import sys
import time
import re
from qiskit.visualization.utils import _validate_input_state
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def iplot_state_hinton(rho, figsize=None):
    """ Create a hinton representation.

        Graphical representation of the input array using a 2D city style
        graph (hinton).

        Args:
            rho (array): Density matrix
            figsize (tuple): Figure size in pixels.

        Example:
            .. code-block::

                from qiskit import QuantumCircuit, BasicAer, execute
                from qiskit.visualization import iplot_state_hinton
                %matplotlib inline

                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])

                backend = BasicAer.get_backend('statevector_simulator')
                job = execute(qc, backend).result()
                iplot_state_hinton(job.get_statevector(qc))

    """

    # HTML
    html_template = Template("""
    <p>
        <div id="hinton_$divNumber"></div>
    </p>
    """)

    # JavaScript
    javascript_template = Template("""
    <script>
        requirejs.config({
            paths: {
                qVisualization: "https://qvisualization.mybluemix.net/q-visualizations"
            }
        });

        require(["qVisualization"], function(qVisualizations) {
            qVisualizations.plotState("hinton_$divNumber",
                                      "hinton",
                                      $executions,
                                      $options);
        });
    </script>
    """)
    rho = _validate_input_state(rho)
    if figsize is None:
        options = {}
    else:
        options = {'width': figsize[0], 'height': figsize[1]}

    # Process data and execute
    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    # Process data and execute
    real = []
    imag = []
    for xvalue in rho:
        row_real = []
        col_imag = []

        for value_real in xvalue.real:
            row_real.append(float(value_real))
        real.append(row_real)

        for value_imag in xvalue.imag:
            col_imag.append(float(value_imag))
        imag.append(col_imag)

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': [{'data': real}, {'data': imag}],
        'options': options
    })

    display(HTML(html + javascript))
