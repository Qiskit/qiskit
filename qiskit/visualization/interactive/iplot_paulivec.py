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
Paulivec visualization
"""
from string import Template
import sys
import time
import re
import numpy as np
from qiskit.quantum_info.operators.pauli import pauli_group
from qiskit.visualization.utils import _validate_input_state
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def process_data(rho):
    """ Sort rho data """
    result = dict()

    num = int(np.log2(len(rho)))
    labels = list(map(lambda x: x.to_label(), pauli_group(num)))
    values = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                      pauli_group(num)))

    for position, label in enumerate(labels):
        result[label] = values[position]
    return result


def iplot_state_paulivec(rho, figsize=None, slider=False, show_legend=False):
    """ Create a paulivec representation.

        Graphical representation of the input array.

        Args:
            rho (array): State vector or density matrix.
            figsize (tuple): Figure size in pixels.
            slider (bool): activate slider
            show_legend (bool): show legend of graph content

        Example:
            .. code-block::

                from qiskit import QuantumCircuit, BasicAer, execute
                from qiskit.visualization import iplot_state_paulivec
                %matplotlib inline

                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])

                backend = BasicAer.get_backend('statevector_simulator')
                job = execute(qc, backend).result()
                iplot_state_paulivec(job.get_statevector(qc))
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="paulivec_$divNumber"></div>
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
            qVisualizations.plotState("paulivec_$divNumber",
                                      "paulivec",
                                      $executions,
                                      $options);
        });
    </script>
    """)
    rho = _validate_input_state(rho)
    # set default figure size if none given
    if figsize is None:
        figsize = (7, 5)

    options = {'width': figsize[0], 'height': figsize[1],
               'slider': int(slider), 'show_legend': int(show_legend)}

    # Process data and execute
    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    data_to_plot = []
    rho_data = process_data(rho)
    data_to_plot.append(dict(
        data=rho_data
    ))

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': data_to_plot,
        'options': options
    })

    display(HTML(html + javascript))
