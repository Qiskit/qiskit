# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Histogram visualization
"""
from string import Template
from collections import Counter
import sys
import time
import re
import numpy as np
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


def iplot_histogram(executions_results, options=None):
    """ Create a histogram representation.

        Graphical representation of the input array using a vertical bars
        style graph.

        Args:
            executions_results (array): Array of dictionaries containing
                    - data (dict): values to represent (ex. {'001' : 130})
                    - name (string): name to show in the legend
                    - device (string): Could be 'real' or 'simulated'
            options (dict): Representation settings containing
                    - width (integer): graph horizontal size
                    - height (integer): graph vertical size
                    - slider (bool): activate slider
                    - number_to_keep (integer): groups max values
                    - show_legend (bool): show legend of graph content
                    - sort (string): Could be 'asc' or 'desc'
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

    if not options:
        options = {}

    if 'slider' in options and options['slider'] is True:
        options['slider'] = 1
    else:
        options['slider'] = 0

    if 'show_legend' in options and options['show_legend'] is False:
        options['show_legend'] = 0
    else:
        options['show_legend'] = 1

    if 'number_to_keep' not in options:
        options['number_to_keep'] = 0

    data_to_plot = []
    for execution in executions_results:
        data = process_data(execution['data'], options['number_to_keep'])
        data_to_plot.append({'data': data})

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'divNumber': div_number,
        'executions': data_to_plot,
        'options': options
    })

    display(HTML(html + javascript))
