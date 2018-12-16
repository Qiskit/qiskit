# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A module for visualizing device coupling maps"""

import matplotlib.pyplot as plt  # pylint: disable=import-error
import matplotlib.patches as mpatches  # pylint: disable=import-error
from qiskit.qiskiterror import QISKitError


class _GraphDist():
    """Transform the circles properly for non-square axes.
    """
    def __init__(self, size, ax, x=True):
        self.size = size
        self.ax = ax  # pylint: disable=invalid-name
        self.x = x

    @property
    def dist_real(self):
        """Compute distance.
        """
        x0, y0 = self.ax.transAxes.transform(  # pylint: disable=invalid-name
            (0, 0))
        x1, y1 = self.ax.transAxes.transform(  # pylint: disable=invalid-name
            (1, 1))
        value = x1 - x0 if self.x else y1 - y0
        return value

    @property
    def dist_abs(self):
        """Distance abs
        """
        bounds = self.ax.get_xlim() if self.x else self.ax.get_ylim()
        return bounds[0] - bounds[1]

    @property
    def value(self):
        """Return value.
        """
        return (self.size / self.dist_real) * self.dist_abs

    def __mul__(self, obj):
        return self.value * obj


def plot_gate_map(backend, figsize=None,
                  plot_directed=False,
                  label_qubits=True,
                  qubit_size=24,
                  line_width=4,
                  font_size=12,
                  qubit_color=None,
                  line_color=None,
                  font_color='w'):
    """Plots the gate map of a device.

    Args:
        backend (BaseBackend): A backend instance,
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        QISKitError: Tried to pass a simulator.
    """
    if backend.configuration().simulator:
        raise QISKitError('Requires a device backend, not simulator.')

    mpl_data = {}

    mpl_data['ibmq_20_tokyo'] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                                 [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
                                 [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
                                 [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]

    mpl_data['ibmq_poughkeepsie'] = mpl_data['ibmq_20_tokyo']

    mpl_data['ibmq_16_melbourne'] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                                     [0, 5], [0, 6], [1, 7], [1, 6], [1, 5],
                                     [1, 4], [1, 3], [1, 2], [1, 1]]

    mpl_data['ibmq_16_rueschlikon'] = [[1, 0], [0, 0], [0, 1], [0, 2], [0, 3],
                                       [0, 4], [0, 5], [0, 6], [0, 7], [1, 7],
                                       [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1]]

    mpl_data['ibmq_5_tenerife'] = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]

    mpl_data['ibmq_5_yorktown'] = mpl_data['ibmq_5_tenerife']

    config = backend.configuration()
    name = config.backend_name
    cmap = config.coupling_map

    dep_names = {'ibmqx5': 'ibmq_16_rueschlikon',
                 'ibmqx4': 'ibmq_5_tenerife',
                 'ibmqx2': 'ibmq_5_yorktown'}

    if name in dep_names.keys():
        name = dep_names[name]

    if name in mpl_data.keys():
        grid_data = mpl_data[name]
    else:
        fig, ax = plt.subplots(figsize=(5, 5))  # pylint: disable=invalid-name
        ax.axis('off')
        return fig

    x_max = max([d[1] for d in grid_data])
    y_max = max([d[0] for d in grid_data])
    max_dim = max(x_max, y_max)

    if figsize is None:
        if x_max/max_dim > 0.33 and y_max/max_dim > 0.33:
            figsize = (5, 5)
        else:
            figsize = (9, 3)

    fig, ax = plt.subplots(figsize=figsize)  # pylint: disable=invalid-name
    ax.axis('off')
    fig.tight_layout()

    # set coloring
    if qubit_color is None:
        qubit_color = ['#648fff']*config.n_qubits
    if line_color is None:
        line_color = ['#648fff']*len(cmap)

    # Add lines for couplings
    for ind, edge in enumerate(cmap):
        is_symmetric = False
        if edge[::-1] in cmap:
            is_symmetric = True
        y_start = grid_data[edge[0]][0]
        x_start = grid_data[edge[0]][1]
        y_end = grid_data[edge[1]][0]
        x_end = grid_data[edge[1]][1]

        if is_symmetric:
            if y_start == y_end:
                x_end = (x_end - x_start)/2+x_start

            elif x_start == x_end:
                y_end = (y_end - y_start)/2+y_start

            else:
                x_end = (x_end - x_start)/2+x_start
                y_end = (y_end - y_start)/2+y_start
        ax.add_artist(plt.Line2D([x_start, x_end], [-y_start, -y_end],
                                 color=line_color[ind], linewidth=line_width,
                                 zorder=0))
        if plot_directed:
            dx = x_end-x_start  # pylint: disable=invalid-name
            dy = y_end-y_start  # pylint: disable=invalid-name
            if is_symmetric:
                x_arrow = x_start+dx*0.95
                y_arrow = -y_start-dy*0.95
                dx_arrow = dx*0.01
                dy_arrow = -dy*0.01
                head_width = 0.15
            else:
                x_arrow = x_start+dx*0.5
                y_arrow = -y_start-dy*0.5
                dx_arrow = dx*0.2
                dy_arrow = -dy*0.2
                head_width = 0.2
            ax.add_patch(mpatches.FancyArrow(x_arrow,
                                             y_arrow,
                                             dx_arrow,
                                             dy_arrow,
                                             head_width=head_width,
                                             length_includes_head=True,
                                             edgecolor=None,
                                             linewidth=0,
                                             facecolor=line_color[ind],
                                             zorder=1))

    # Add circles for qubits
    for var, idx in enumerate(grid_data):
        _idx = [idx[1], -idx[0]]
        width = _GraphDist(qubit_size, ax, True)
        height = _GraphDist(qubit_size, ax, False)
        ax.add_artist(mpatches.Ellipse(
            _idx, width, height, color=qubit_color[var], zorder=1))
        if label_qubits:
            ax.text(*_idx, s=str(var),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color=font_color, size=font_size, weight='bold')
    ax.set_xlim([-1, x_max+1])
    ax.set_ylim([-(y_max+1), 1])
    plt.close(fig)
    return fig
