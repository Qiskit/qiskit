# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Bloch sphere visualization
"""
from qiskit.tools.qi.pauli import pauli_group, pauli_singles
from IPython.core.display import display, HTML
from string import Template
import numpy as np
import time
import re


def iplot_blochsphere(rho, options={}):
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

    # Process data and execute
    num = int(np.log2(len(rho)))

    bloch_data = []
    for i in range(num):
        bloch_state = list(map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), rho))),
                               pauli_singles(i, num)))
        bloch_data.append(bloch_state)

    divNumber = str(time.time())
    divNumber = re.sub('[.]', '', divNumber)

    html = html_template.substitute({
        'divNumber': divNumber
    })

    javascript = javascript_template.substitute({
        'data': bloch_data,
        'divNumber': divNumber,
        'options': options
    })

    display(HTML(html + javascript))
