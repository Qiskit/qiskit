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
Histogram visualization
"""
from string import Template
from collections import Counter
import sys
import time
import re
import numpy as np
from ..exceptions import VisualizationError

if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def process_data(data, number_to_keep):
    """ Prepare received data for representation.

        Args:
            data (dict): values to represent (ex. {'001' : 130})
            number_to_keep (int): number of elements to show individually.

        Returns:
            dict: processed data to show.
    """

    result = dict()

    if number_to_keep != 0:
        data_temp = dict(Counter(data).most_common(number_to_keep))
        data_temp['rest'] = sum(data.values()) - sum(data_temp.values())
        data = data_temp

    labels = data
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    for position, label in enumerate(labels):
        result[label] = round(pvalues[position], 5)

    return result


def iplot_histogram(data, figsize=None, number_to_keep=None,
                    sort='asc', legend=None):
    """ Create a histogram representation.

        Graphical representation of the input array using a vertical bars
        style graph.

        Args:
            data (list or dict):  This is either a list of dicts or a single
                dict containing the values to represent (ex. {'001' : 130})
            figsize (tuple): Figure size in pixels.
            number_to_keep (int): The number of terms to plot and
                rest is made into a single bar called other values
            sort (string): Could be 'asc' or 'desc'
            legend (list): A list of strings to use for labels of the data.
                The number of entries must match the length of data.

        Raises:
            VisualizationError: When legend is provided and the length doesn't
                match the input data.

        Example:
            .. code-block::

                from qiskit import QuantumCircuit, BasicAer, execute
                from qiskit.visualization import iplot_histogram
                %matplotlib inline

                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])

                backend = BasicAer.get_backend('qasm_simulator')
                job = execute(qc, backend)
                iplot_histogram(job.result().get_counts())
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="histogram_$divNumber"></div>
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
            qVisualizations.plotState("histogram_$divNumber",
                                      "histogram",
                                      $executions,
                                      $options);
        });
    </script>
    """)

    # Process data and execute
    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    # set default figure size if none provided
    if figsize is None:
        figsize = (7, 5)

    options = {'number_to_keep': 0 if number_to_keep is None else number_to_keep,
               'sort': sort,
               'show_legend': 0,
               'width': int(figsize[0]),
               'height': int(figsize[1])}
    if legend:
        options['show_legend'] = 1

    data_to_plot = []
    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise VisualizationError("Length of legendL (%s) doesn't match number "
                                 "of input executions: %s" %
                                 (len(legend), len(data)))

    for item, execution in enumerate(data):
        exec_data = process_data(execution, options['number_to_keep'])
        out_dict = {'data': exec_data}
        if legend:
            out_dict['name'] = legend[item]
        data_to_plot.append(out_dict)

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': data_to_plot,
        'options': options
    })

    display(HTML(html + javascript))
