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
Bloch sphere visualization
"""
from string import Template
import sys
import time
import re
import numpy as np
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.visualization.utils import _validate_input_state
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

        Example:
            .. code-block::

                from qiskit import QuantumCircuit, BasicAer, execute
                from qiskit.visualization import iplot_bloch_multivector
                %matplotlib inline

                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])

                backend = BasicAer.get_backend('statevector_simulator')
                job = execute(qc, backend).result()
                iplot_bloch_multivector(job.get_statevector(qc))
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
