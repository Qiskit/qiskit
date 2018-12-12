# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Bloch sphere visualization
"""
from string import Template
import sys
import time
import re
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools.visualization._utils import _validate_input_state
if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def iplot_bloch_multivector(rho, figsize=None):
    """ Create a bloch sphere representation.

        Graphical representation of the input array, using as much bloch
        spheres as qubit are required.

        Args:
            rho (array): State vector or density matrix
            figsize (tuple): Figure size in pixels.
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="content_$divNumber" style="position: absolute; z-index: 1;">
            <div id="bloch_$divNumber"></div>
        </div>
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
        data = $data;
        dataValues = [];
        for (var i = 0; i < data.length; i++) {
            // Coordinates
            var x = data[i][0];
            var y = data[i][1];
            var z = data[i][2];
            var point = {'x': x,
                        'y': y,
                        'z': z};
            dataValues.push(point);
        }

        require(["qVisualization"], function(qVisualizations) {
            // Plot figure
            qVisualizations.plotState("bloch_$divNumber",
                                      "bloch",
                                      dataValues,
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
    num = int(np.log2(len(rho)))

    bloch_data = []
    for i in range(num):
        pauli_singles = [Pauli.pauli_single(num, i, 'X'), Pauli.pauli_single(num, i, 'Y'),
                         Pauli.pauli_single(num, i, 'Z')]
        bloch_state = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                               pauli_singles))
        bloch_data.append(bloch_state)

    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'data': bloch_data,
        'divNumber': div_number,
        'options': options
    })

    display(HTML(html + javascript))
