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
Visualization functions for measurement counts.
"""

from collections import Counter, OrderedDict
import functools
import numpy as np

from qiskit.exceptions import MissingOptionalLibraryError
from .matplotlib import HAS_MATPLOTLIB
from .exceptions import VisualizationError


def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two bit strings

    Args:
        str1 (str): First string.
        str2 (str): Second string.
    Returns:
        int: Distance between strings.
    Raises:
        VisualizationError: Strings not same length
    """
    if len(str1) != len(str2):
        raise VisualizationError("Strings not same length.")
    return sum(s1 != s2 for s1, s2 in zip(str1, str2))


VALID_SORTS = ["asc", "desc", "hamming", "value", "value_desc"]
DIST_MEAS = {"hamming": hamming_distance}


def plot_histogram(
    data,
    figsize=(7, 5),
    color=None,
    number_to_keep=None,
    sort="asc",
    target_string=None,
    legend=None,
    bar_labels=True,
    title=None,
    ax=None,
):
    """Plot a histogram of data.

    Args:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        figsize (tuple): Figure size in inches.
        color (list or str): String or list of strings for histogram bar colors.
        number_to_keep (int): The number of terms to plot and rest
            is made into a single bar called 'rest'.
        sort (string): Could be `'asc'`, `'desc'`, `'hamming'`, `'value'`, or
            `'value_desc'`. If set to `'value'` or `'value_desc'` the x axis
            will be sorted by the maximum probability for each bitstring.
            Defaults to `'asc'`.
        target_string (str): Target string if 'sort' is a distance measure.
        legend(list): A list of strings to use for labels of the data.
            The number of entries must match the length of data (if data is a
            list or 1 if it's a dict)
        bar_labels (bool): Label each bar in histogram with probability value.
        title (str): A string to use for the plot title
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.

    Returns:
        matplotlib.Figure:
            A figure for the rendered histogram, if the ``ax``
            kwarg is not set.

    Raises:
        MissingOptionalLibraryError: Matplotlib not available.
        VisualizationError: When legend is provided and the length doesn't
            match the input data.

    Example:
        .. jupyter-execute::

           from qiskit import QuantumCircuit, BasicAer, execute
           from qiskit.visualization import plot_histogram
           %matplotlib inline

           qc = QuantumCircuit(2, 2)
           qc.h(0)
           qc.cx(0, 1)
           qc.measure([0, 1], [0, 1])

           backend = BasicAer.get_backend('qasm_simulator')
           job = execute(qc, backend)
           plot_histogram(job.result().get_counts(), color='midnightblue', title="New Histogram")
    """
    if not HAS_MATPLOTLIB:
        raise MissingOptionalLibraryError(
            libname="Matplotlib",
            name="plot_histogram",
            pip_install="pip install matplotlib",
        )
    from matplotlib import get_backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if sort not in VALID_SORTS:
        raise VisualizationError(
            "Value of sort option, %s, isn't a "
            "valid choice. Must be 'asc', "
            "'desc', 'hamming', 'value', 'value_desc'"
        )
    if sort in DIST_MEAS.keys() and target_string is None:
        err_msg = "Must define target_string when using distance measure."
        raise VisualizationError(err_msg)

    if isinstance(data, dict):
        data = [data]

    if legend and len(legend) != len(data):
        raise VisualizationError(
            "Length of legendL (%s) doesn't match "
            "number of input executions: %s" % (len(legend), len(data))
        )

    # Set bar colors
    if color is None:
        color = ["#648fff", "#dc267f", "#785ef0", "#ffb000", "#fe6100"]
    elif isinstance(color, str):
        color = [color]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    labels = list(sorted(functools.reduce(lambda x, y: x.union(y.keys()), data, set())))
    if number_to_keep is not None:
        labels.append("rest")

    if sort in DIST_MEAS.keys():
        dist = []
        for item in labels:
            dist.append(DIST_MEAS[sort](item, target_string))

        labels = [list(x) for x in zip(*sorted(zip(dist, labels), key=lambda pair: pair[0]))][1]
    elif "value" in sort:
        combined_counts = {}
        if isinstance(data, dict):
            combined_counts = data
        else:
            for counts in data:
                for count in counts:
                    prev_count = combined_counts.get(count, 0)
                    combined_counts[count] = max(prev_count, counts[count])
        labels = list(sorted(combined_counts.keys(), key=lambda key: combined_counts[key]))

    length = len(data)
    width = 1 / (len(data) + 1)  # the width of the bars

    labels_dict, all_pvalues, all_inds = _plot_histogram_data(data, labels, number_to_keep)
    rects = []
    for item, _ in enumerate(data):
        for idx, val in enumerate(all_pvalues[item]):
            label = None
            if not idx and legend:
                label = legend[item]
            if val >= 0:
                rects.append(
                    ax.bar(
                        idx + item * width,
                        val,
                        width,
                        label=label,
                        color=color[item % len(color)],
                        zorder=2,
                    )
                )
        bar_center = (width / 2) * (length - 1)
        ax.set_xticks(all_inds[item] + bar_center)
        ax.set_xticklabels(labels_dict.keys(), fontsize=14, rotation=70)
        # attach some text labels
        if bar_labels:
            for rect in rects:
                for rec in rect:
                    height = rec.get_height()
                    if height >= 1e-3:
                        ax.text(
                            rec.get_x() + rec.get_width() / 2.0,
                            1.05 * height,
                            "%.3f" % float(height),
                            ha="center",
                            va="bottom",
                            zorder=3,
                        )
                    else:
                        ax.text(
                            rec.get_x() + rec.get_width() / 2.0,
                            1.05 * height,
                            "0",
                            ha="center",
                            va="bottom",
                            zorder=3,
                        )

    # add some text for labels, title, and axes ticks
    ax.set_ylabel("Probabilities", fontsize=14)
    all_vals = np.concatenate(all_pvalues).ravel()
    ax.set_ylim([0.0, min([1.2, max(1.2 * val for val in all_vals)])])
    if "desc" in sort:
        ax.invert_xaxis()

    ax.yaxis.set_major_locator(MaxNLocator(5))
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    plt.grid(which="major", axis="y", zorder=0, linestyle="--")
    if title:
        plt.title(title)

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            borderaxespad=0,
            frameon=True,
            fontsize=12,
        )
    if fig:
        if get_backend() in ["module://ipykernel.pylab.backend_inline", "nbAgg"]:
            plt.close(fig)
    return fig


def _plot_histogram_data(data, labels, number_to_keep):
    """Generate the data needed for plotting counts.

    Parameters:
        data (list or dict): This is either a list of dictionaries or a single
            dict containing the values to represent (ex {'001': 130})
        labels (list): The list of bitstring labels for the plot.
        number_to_keep (int): The number of terms to plot and rest
            is made into a single bar called 'rest'.

    Returns:
        tuple: tuple containing:
            (dict): The labels actually used in the plotting.
            (list): List of ndarrays for the bars in each experiment.
            (list): Indices for the locations of the bars for each
                    experiment.
    """
    labels_dict = OrderedDict()

    all_pvalues = []
    all_inds = []
    for execution in data:
        if number_to_keep is not None:
            data_temp = dict(Counter(execution).most_common(number_to_keep))
            data_temp["rest"] = sum(execution.values()) - sum(data_temp.values())
            execution = data_temp
        values = []
        for key in labels:
            if key not in execution:
                if number_to_keep is None:
                    labels_dict[key] = 1
                    values.append(0)
                else:
                    values.append(-1)
            else:
                labels_dict[key] = 1
                values.append(execution[key])
        values = np.array(values, dtype=float)
        pvalues = values / sum(values)
        all_pvalues.append(pvalues)
        numelem = len(values)
        ind = np.arange(numelem)  # the x locations for the groups
        all_inds.append(ind)

    return labels_dict, all_pvalues, all_inds
