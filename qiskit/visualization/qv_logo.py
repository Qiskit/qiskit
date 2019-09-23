# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""A module for plotting the Quantum Volume logo"""

import numpy as np
from .matplotlib import HAS_MATPLOTLIB
from .exceptions import VisualizationError

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from matplotlib import get_backend


def plot_qv_logo(qv,
                 cube_edge=4,
                 fontcolor='k',
                 fontsize=96,
                 figsize=(8, 8),
                 ax=None,
                 fontname=None):

    """Plot the QV cube for a given QV value <= 512.

    Args:
        qv (int): Quantum volume.
        cube_edge (int): Length of cube edge, 4 or 8.
        fontcolor (str): Color of font, 'k' or 'w'.
        fontsize (int): Fontsize.
        figsize (tuple): Figure size in inches.
        ax (Axes3D): Optional input 3d axes.
        fontname (str): Optional font name, e.g. 'IBM Plex Mono'

    Returns:
        Figure: Matplotlib figure instance.

    Raises:
        VisualizationError: Incorrect inputs.
    """

    def explode(data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    if cube_edge not in [4, 8]:
        raise VisualizationError('cube_size must be 4 or 8.')

    if not np.log2(qv).is_integer():
        raise VisualizationError('Invalid QV value.')

    if qv > 512:
        raise VisualizationError('QV value is too large.')

    # build up the numpy logo
    n_voxels = np.zeros((cube_edge,
                         cube_edge,
                         cube_edge), dtype=bool)
    if qv == 4:
        n_voxels[0:2, 0:2, 0:1] = True
    elif qv == 8:
        n_voxels[0:2, 0:2, 0:2] = True
    elif qv == 16:
        n_voxels[0:4, 0:2, 0:2] = True
    elif qv == 32:
        n_voxels[0:4, 0:4, 0:2] = True
    elif qv == 64:
        n_voxels[0:4, 0:4, 0:4] = True
    else:
        if cube_edge == 4:
            raise VisualizationError('Must use larger cube_size')

    if cube_edge != 8 and qv > 64:
        raise VisualizationError('cube_size must be 8.')

    if qv == 128:
        n_voxels[0:8, 0:4, 0:4] = True
    elif qv == 256:
        n_voxels[0:8, 0:8, 0:4] = True
    elif qv == 512:
        n_voxels[0:8, 0:8, 0:8] = True

    facecolors = np.where(n_voxels, '#ee538b', '#ffffff20')
    edgecolors = np.where(n_voxels, '#f2f4f830', '#f2f4f830')
    filled = np.ones(n_voxels.shape)

    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    given_ax = False
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
    else:
        given_ax = True
        if ax.name != '3d':
            raise VisualizationError("Input axes must be '3d'.")

    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    ax.set_axis_off()
    ax.set_title('QV{}'.format(qv),
                 fontsize=fontsize,
                 color=fontcolor,
                 fontname=fontname)
    if not given_ax:
        if get_backend() in ['module://ipykernel.pylab.backend_inline',
                             'nbAgg']:
            plt.close(fig)
        return fig
    return None
