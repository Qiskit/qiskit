# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from abc import abstractmethod
from typing import List, Union, Tuple
import os
import csv

import numpy as np


class VarQteErrorCalculator:
    @abstractmethod
    def _get_error_bound(
        self, gradient_errors: List, time: List, *args
    ) -> Union[List, Tuple[List, List]]:
        """
        Get the upper bound to a global phase agnostic l2-norm error for VarQTE simulation
        Args:
            gradient_errors: Error of the state propagation gradient for each t in times
            time: List of all points in time considered throughout the simulation
            args: Optional parameters for the error bound
        Returns:
            List of the error upper bound for all times
        Raises: NotImplementedError
        """
        raise NotImplementedError

    def error_bound(
        self, data_dir: str, imag_reverse_bound: bool = False
    ) -> Union[List, Tuple[List, List]]:
        """
        Evaluate an upper bound to the error of the VarQTE simulation
        Args:
            data_dir: Directory where the snapshots were stored
            imag_reverse_bound: Compute the additional reverse bound (ignored if notVarQITE)
        Returns: Error bounds for all accessed time points in the evolution
        """
        # Read data
        with open(os.path.join(data_dir, "varqte_output.csv"), mode="r") as csv_file:
            fieldnames = [
                "t",
                "params",
                "num_params",
                "num_time_steps",
                "error_bound",
                "error_bound_grad",
                "error_grad",
                "resid",
                "fidelity",
                "true_error",
                "phase_agnostic_true_error",
                "true_to_euler_error",
                "trained_to_euler_error",
                "target_energy",
                "trained_energy",
                "energy_error",
                "h_norm",
                "h_squared",
                "h_trip",
                "variance",
                "dtdt_trained",
                "re_im_grad",
            ]
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            first = True
            error_bound = []
            error_bound_grad = []
            grad_errors = []
            energies = []
            time = []
            # time_steps = []
            h_squareds = []
            h_trips = []
            stddevs = []

            for line in reader:
                if first:
                    first = False
                    continue
                t_line = float(line["t"])
                if t_line in time:
                    continue
                time.append(t_line)
                error_bound.append(float(line["error_bound"]))
                error_bound_grad.append(float(line["error_bound_grad"]))
                grad_errors.append(float(line["error_grad"]))
                energies.append(float(line["trained_energy"]))
                h_squareds.append(float(line["h_squared"]))
                try:
                    h_trips.append(float(line["h_trip"]))
                except Exception:
                    pass
                stddevs.append(np.sqrt(float(line["variance"])))

        if not np.iscomplex(self._operator.coeff):
            direct_error_bounds = self._get_error_bound(
                grad_errors, time, stddevs, h_squareds, h_trips, energies, trapezoidal=False
            )
            print("direct errors", direct_error_bounds)
            np.save(os.path.join(data_dir, "direct_error_bounds.npy"), direct_error_bounds)

            if imag_reverse_bound:
                imag_reverse_bound = self._get_reverse_error_bound(time, error_bound_grad, stddevs)
                return error_bound, imag_reverse_bound

        return error_bound
