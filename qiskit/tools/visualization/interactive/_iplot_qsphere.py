# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qsphere visualization
"""
from qiskit.tools.qi.pauli import pauli_group, pauli_singles
from scipy import linalg
from functools import reduce
from IPython.core.display import display, HTML
from string import Template
import numpy as np
import time
import re


def iplot_qsphere(rho, options={}):
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
        data = $data;
        require(["qVisualization"], function(qVisualizations) {
            qVisualizations.plotState("qsphere_$divNumber",
                                      "qsphere",
                                      data,
                                      $options);
        });
    </script>

    """)
    qspheres_data = []
    # Process data and execute
    num = int(np.log2(len(rho)))

    # get the eigenvectors and egivenvalues
    we, stateall = linalg.eigh(rho)

    for k in range(2**num):
        # start with the max
        probmix = we.max()
        prob_location = we.argmax()
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

                qpoint = {
                    'x': xvalue,
                    'y': yvalue,
                    'z': zvalue,
                    'prob': prob,
                    'phase': angle
                }
                spherepoints.append(qpoint)

            # Associate all points to one sphere
            sphere = {
                'points': spherepoints,
                'eigenvalue': probmix
            }

            # Add sphere to the spheres array
            qspheres_data.append(sphere)
            we[prob_location] = 0

    divNumber = str(time.time())
    divNumber = re.sub('[.]', '', divNumber)

    html = html_template.substitute({
        'divNumber': divNumber
    })

    javascript = javascript_template.substitute({
        'data': qspheres_data,
        'divNumber': divNumber,
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


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    assert s.count("0") == n - k, "s must be a string of 0 and 1"
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def lex_index(n, k, lst):
    """Return  the lex index of a combination..

    Args:
        n (int): the total number of options .
        k (int): The number of elements.
        lst (list): list

    Returns:
        int: returns int index for lex order

    """
    assert len(lst) == k, "list should have length k"
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum([n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)])
    return int(dualm)
