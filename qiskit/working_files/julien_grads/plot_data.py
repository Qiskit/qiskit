"""Plot the convergences."""

import warnings
import numpy as np
from matplotlib import pyplot as plt


def color_cycle():
    """Get the color cycle for this project."""
    return ['tab:blue', 'tab:purple', 'tab:pink', 'tab:red']


def plot_data(data, saveas=None, plot_kwargs=None):
    """Plot the convergence histories of the methods in ``self.data``."""

    labels = list(data.keys())
    print('labels', labels)

    if plot_kwargs is None:
        plot_kwargs = {label: {} for label in labels}
    else:
        for label in labels:
            if label not in plot_kwargs.keys():
                plot_kwargs[label] = {}

    colors = color_cycle()

    # if data has just a single run it might not be wrapped in a list
    for label in labels:
        if not isinstance(data[label], list):
            # replace by list
            data[label] = [data[label]]

    for k, (label, histories) in enumerate(data.items()):
        print(label)
        print(histories)
        kwargs = plot_kwargs[label]
        if 'color' not in kwargs.keys() and 'c' not in kwargs.keys():
            kwargs['color'] = colors[k]

        stepsizes = []
        losses = []
        circuit_evals = histories[0].get('nfevs', None)  # are the same for all runs

        for history in histories:
            losses.append(history.get('fx', None))
            stepsizes.append(history.get('stepsizes', None))

        # plot loss
        if None in losses:
            warnings.warn('The `fx` key was missing, not plotting the losses.')
        else:
            means = np.mean(losses, axis=0)
            std = np.std(losses, axis=0)

            for x, fignum in zip([circuit_evals, np.arange(len(circuit_evals))],
                                 [1, 2]):
                plt.figure(fignum)
                plt.plot(x, means, label=label, **kwargs)

                # only plot stddev if we have more than 1 run
                if len(losses) > 1:
                    plt.fill_between(x, means - std, means + std,
                                     alpha=0.2, **kwargs)

        if None in stepsizes:
            warnings.warn('The `stepsizes` key was missing, not plotting the stepsizes.')
        else:
            # plot stepsizes
            means = np.mean(stepsizes, axis=0)
            std = np.std(stepsizes, axis=0)
            x = np.arange(len(circuit_evals) - 1)

            plt.figure(3)
            plt.plot(x, means, label=labels[k], **kwargs)

            # only plot stddev if we have more than 1 run
            if len(stepsizes) > 1:
                plt.fill_between(x, means - std, means + std,
                                 alpha=0.2, **kwargs)

    # set labels and save
    if saveas is not None:
        plt.figure(1)
        plt.xlabel('#circuits')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(saveas + '_ccts.pdf')
        plt.savefig(saveas + '_ccts.png')

        plt.figure(2)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(saveas + '_it.pdf')
        plt.savefig(saveas + '_it.png')

        plt.figure(3)
        plt.xlabel('iterations')
        plt.ylabel('stepsize')
        plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.savefig(saveas + '_steps.pdf')
        plt.savefig(saveas + '_steps.png')
