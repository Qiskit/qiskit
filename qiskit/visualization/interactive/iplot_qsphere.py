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
Qsphere visualization
"""

import re
import sys
import time
from functools import reduce
from string import Template

import numpy as np
from scipy import linalg
from qiskit.visualization.utils import _validate_input_state
from qiskit.visualization.exceptions import VisualizationError


if ('ipykernel' in sys.modules) and ('spyder' not in sys.modules):
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print("Error importing IPython.core.display")


def iplot_state_qsphere(rho, figsize=None):
    """ Create a Q sphere representation.

        Graphical representation of the input array, using a Q sphere for each
        eigenvalue.

        Args:
            rho (array): State vector or density matrix.
            figsize (tuple): Figure size in pixels.

        Example:
            .. code-block::

                from qiskit import QuantumCircuit, BasicAer, execute
                from qiskit.visualization import iplot_state_qsphere
                %matplotlib inline

                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure([0, 1], [0, 1])

                backend = BasicAer.get_backend('statevector_simulator')
                job = execute(qc, backend).result()
                iplot_state_qsphere(job.get_statevector(qc))
    """

    # HTML
    html_template = Template("""
    <p>
        <div id="content_$divNumber" style="position: absolute; z-index: 1;">
            <div id="qsphere_$divNumber"></div>
        </div>
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
            data = $data;
            qVisualizations.plotState("qsphere_$divNumber",
                                      "qsphere",
                                      data,
                                      $options);
        });
    </script>

    """)
    rho = _validate_input_state(rho)
    if figsize is None:
        options = {}
    else:
        options = {'width': figsize[0], 'height': figsize[1]}

    qspheres_data = []
    # Process data and execute
    num = int(np.log2(len(rho)))

    # get the eigenvectors and eigenvalues
    weig, stateall = linalg.eigh(rho)

    for _ in range(2**num):
        # start with the max
        probmix = weig.max()
        prob_location = weig.argmax()
        if probmix > 0.001:
            # print("The " + str(k) + "th eigenvalue = " + str(probmix))
            # get the max eigenvalue
            state = stateall[:, prob_location]
            loc = np.absolute(state).argmax()
            # get the element location closes to lowest bin representation.
            for j in range(2**num):
                test = np.absolute(np.absolute(state[j]) -
                                   np.absolute(state[loc]))
                if test < 0.001:
                    loc = j
                    break
            # remove the global phase
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j*angles)
            state = angleset*state
            state.flatten()

            spherepoints = []
            for i in range(2**num):
                # get x,y,z points

                element = bin(i)[2:].zfill(num)
                weight = element.count("1")

                number_of_divisions = n_choose_k(num, weight)
                weight_order = bit_string_index(element)

                angle = weight_order * 2 * np.pi / number_of_divisions

                zvalue = -2 * weight / num + 1
                xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)

                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                angles = (np.angle(state[i]) + 2 * np.pi) % (2 * np.pi)
                qpoint = {
                    'x': xvalue,
                    'y': yvalue,
                    'z': zvalue,
                    'prob': prob,
                    'phase': angles
                }
                spherepoints.append(qpoint)

            # Associate all points to one sphere
            sphere = {
                'points': spherepoints,
                'eigenvalue': probmix
            }

            # Add sphere to the spheres array
            qspheres_data.append(sphere)
            weig[prob_location] = 0

    div_number = str(time.time())
    div_number = re.sub('[.]', '', div_number)

    html = html_template.substitute({
        'divNumber': div_number
    })

    javascript = javascript_template.substitute({
        'data': qspheres_data,
        'divNumber': div_number,
        'options': options
    })

    display(HTML(html + javascript))


def n_choose_k(n, k):
    """Return the number of combinations for n choose k.

    Args:
        n (int): the total number of options .
        k (int): The number of elements.

    Returns:
        int: returns the binomial coefficient
    """
    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1],
                  zip(range(n - k + 1, n + 1),
                      range(1, k + 1)), 1)


def bit_string_index(text):
    """Return the index of a string of 0s and 1s."""
    n = len(text)
    k = text.count("1")
    if text.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(text) if char == "1"]
    return lex_index(n, k, ones)


def lex_index(n, k, lst):
    """Return  the lex index of a combination..

    Args:
        n (int): the total number of options .
        k (int): The number of elements.
        lst (list): list

    Returns:
        int: returns int index for lex order

    Raises:
        VisualizationError: if length of list is not equal to k
    """
    if len(lst) != k:
        raise VisualizationError("list should have length k")
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum([n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)])
    return int(dualm)
