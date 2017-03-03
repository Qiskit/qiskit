"""
Basic plotting methods using matplotlib.

These include methods to plot Bloch vectors and histograms, for example.
Author: Andrew Cross, Jay Gambetta
"""


from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


class Arrow3D(FancyArrowPatch):
    """Standard 3D arrow."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Create arrow."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the arrow."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plotRBData(xdata, ydatas, yavg, fit, survival_prob):
    """Plot randomized benchmarking data.

    xdata = list of subsequence lengths
    ydatas = list of lists of survival probabilities for each sequence
    yavg = mean of the survival probabilities at each sequence length
    fit = list of fitting parameters [a, b, alpha]
    survival_prob = function that computes survival probability
    """
    # Plot the result for each sequence
    for ydata in ydatas:
        plt.plot(xdata, ydata, 'rx')
    # Plot the mean
    plt.plot(xdata, yavg, 'bo')
    # Plot the fit
    plt.plot(xdata, survival_prob(xdata, *fit), 'b-')
    plt.show()


def plotBlochVector(bloch, title=""):
    """Plot a Bloch vector.

    Plot a sphere, axes, the Bloch vector, and its projections onto each axis.
    bloch is a 3-tuple (x,y,z)
    title is a string, the plot title
    """
    # Set arrow lengths
    arlen = 1.3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")

    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="b", alpha=0.1)

    # Plot arrows (axes, Bloch vector, its projections)
    xa = Arrow3D([0, arlen], [0, 0], [0, 0], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color="k")
    ya = Arrow3D([0, 0], [0, arlen], [0, 0], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color="k")
    za = Arrow3D([0, 0], [0, 0], [0, arlen], mutation_scale=20, lw=1,
                 arrowstyle="-|>", color="k")
    a = Arrow3D([0, bloch[0]], [0, bloch[1]], [0, bloch[2]], mutation_scale=20,
                lw=2, arrowstyle="simple", color="k")
    bax = Arrow3D([0, bloch[0]], [0, 0], [0, 0], mutation_scale=20, lw=2,
                  arrowstyle="-", color="r")
    bay = Arrow3D([0, 0], [0, bloch[1]], [0, 0], mutation_scale=20, lw=2,
                  arrowstyle="-", color="g")
    baz = Arrow3D([0, 0], [0, 0], [0, bloch[2]], mutation_scale=20, lw=2,
                  arrowstyle="-", color="b")
    arrowlist = [xa, ya, za, a, bax, bay, baz]
    for arr in arrowlist:
        ax.add_artist(arr)

    # Rotate the view
    ax.view_init(30, 30)

    # Annotate the axes, shifts are ad-hoc for this (30,30) view
    xp, yp, _ = proj3d.proj_transform(arlen, 0, 0, ax.get_proj())
    plt.annotate("x", xy=(xp, yp), xytext=(-3, -8),
                 textcoords='offset points', ha='right', va='bottom')
    xp, yp, _ = proj3d.proj_transform(0, arlen, 0, ax.get_proj())
    plt.annotate("y", xy=(xp, yp), xytext=(6, -5),
                 textcoords='offset points', ha='right', va='bottom')
    xp, yp, _ = proj3d.proj_transform(0, 0, arlen, ax.get_proj())
    plt.annotate("z", xy=(xp, yp), xytext=(2, 0),
                 textcoords='offset points', ha='right', va='bottom')

    plt.title(title)
    plt.show()


def plotHistogram(data):
    """Plot a histogram of data."""
    labels = sorted(data)
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    numelem = len(values)
    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects = ax.bar(ind, pvalues, width, color='seagreen')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Probabilities', fontsize=20)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylim([0., min([1.2, max([1.2 * val for val in pvalues])])])
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % float(height),
                ha='center', va='bottom')
    plt.show()
