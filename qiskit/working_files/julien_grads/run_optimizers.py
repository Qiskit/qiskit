"""Run optimizers."""

from time import time
import numpy as np


def run_optimizers(optimizers, objective_fn, initial_point, saveas, seed=None, runs=1):
    """Run all optimizers on the objective function."""

    if not isinstance(optimizers, dict):
        raise ValueError(
            'Please pass the optimizers as dict: {label: optimizer} or '
            '{label: (optimizer, **kwargs)}')

    if seed is not None:
        np.random.seed(seed)

    data = {}

    if isinstance(runs, int):
        runs = len(optimizers) * [runs]

    for nruns, (label, optimizer) in zip(runs, optimizers.items()):
        print(20 * '-')
        print(f'Running {label}...')

        if isinstance(optimizer, tuple):
            optimizer, kwargs = optimizer
        else:
            kwargs = {}

        histories = []
        for run in range(nruns):
            start = time()
            _ = optimizer.optimize(
                len(initial_point), objective_fn, initial_point=initial_point, **kwargs
            )
            print(f'{run + 1}/{nruns} took {time() - start}s')

            histories.append(optimizer.history)

        data[label] = histories
        print()

    np.save(saveas, data)

    return data
