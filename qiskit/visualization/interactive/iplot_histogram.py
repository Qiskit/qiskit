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

import warnings

from qiskit.visualization.counts_visualization import plot_histogram


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
        Returns:
            Figure: A matplotlib figure for the visualization
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
    warnings.warn(
        "The iplot_histogram function is deprecated and will be "
        "removed in a future release. The hosted code this depended on no "
        "longer exists so this is falling back to use the matplotlib "
        "visualizations. qiskit.visualization.plot_histogram should be "
        "used directly moving forward.", DeprecationWarning, stacklevel=2)
    fig = plot_histogram(data, figsize=figsize, number_to_keep=number_to_keep,
                         sort=sort, legend=legend)
    return fig
