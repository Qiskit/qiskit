# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of discrete probability metrics."""
import numpy as np


def hellinger_fidelity(dist_p, dist_q):
    """Computes the Hellinger fidelity between
    two counts distributions.

    The fidelity is defined as 1-H where H is the
    Hellinger distance.  This value is bounded
    in the range [0, 1].

    Parameters:
        dist_p (dict): First dict of counts.
        dist_q (dict): Second dict of counts.

    Returns:
        float: Fidelity

    Example:
        .. jupyter-execute::
            :hide-code:
            :hide-output:

            from qiskit.test.ibmq_mock import mock_get_backend
            mock_get_backend('FakeVigo')

        .. jupyter-execute::

            from qiskit import QuantumCircuit, execute, IBMQ
            from qiskit.quantum_info.analysis import hellinger_fidelity
            from qiskit.providers.aer import noise

            provider = IBMQ.load_account()
            accountProvider = IBMQ.get_provider(hub='ibm-q')
            backend = accountProvider.get_backend('ibmq_vigo')

            sim = Aer.get_backend('qasm_simulator')
            properties = backend.properties()
            coupling_map = backend.configuration().coupling_map

            noise_model = noise.device.basic_device_noise_model(properties)
            basis_gates = noise_model.basis_gates

            ideal_res = execute(qc, sim).result()

            noise_res = execute(qc, sim,
                                coupling_map=coupling_map,
                                noise_model=noise_model,
                                basis_gates=basis_gates).result()

            hellinger_fidelity(ideal_res.get_counts(), noise_res.get_counts())
    """
    p_sum = sum(dist_p.values())
    q_sum = sum(dist_q.values())

    p_normed = {}
    for key, val in dist_p.items():
        p_normed[key] = val/p_sum

    q_normed = {}
    for key, val in dist_q.items():
        q_normed[key] = val/q_sum

    total = 0
    for key, val in p_normed.items():
        if key in q_normed.keys():
            total += (np.sqrt(val) - np.sqrt(q_normed[key]))**2
            del q_normed[key]
        else:
            total += val
    total += sum(q_normed.values())

    dist = np.sqrt(total)/np.sqrt(2)

    return 1-dist
