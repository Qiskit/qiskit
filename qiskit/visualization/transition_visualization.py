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
Visualization function for animation of state transitions by applying gates to single qubit.
"""
from math import sin, cos, acos, sqrt
import numpy as np

try:
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from IPython.display import HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from qiskit.visualization.bloch import Bloch
from qiskit.visualization.exceptions import VisualizationError


def _normalize(v, tolerance=0.00001):
    """Makes sure magnitude of the vector is 1 with given tolerance"""

    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


class _Quaternion:
    """For calculating vectors on unit sphere"""
    def __init__(self):
        self._val = None

    @staticmethod
    def from_axisangle(theta, v):
        v = _normalize(v)

        new_quaternion = _Quaternion()
        new_quaternion._axisangle_to_q(theta, v)
        return new_quaternion

    @staticmethod
    def from_value(value):
        new_quaternion = _Quaternion()
        new_quaternion._val = value
        return new_quaternion

    def _axisangle_to_q(self, theta, v):
        x = v[0]
        y = v[1]
        z = v[2]

        w = cos(theta/2.)
        x = x * sin(theta/2.)
        y = y * sin(theta/2.)
        z = z * sin(theta/2.)

        self._val = np.array([w, x, y, z])

    def __mul__(self, b):

        if isinstance(b, _Quaternion):
            return self._multiply_with_quaternion(b)
        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) != 3:
                raise Exception(f"Input vector has invalid length {len(b)}")
            return self._multiply_with_vector(b)
        else:
            raise Exception(f"Multiplication with unknown type {type(b)}")

    def _multiply_with_quaternion(self, q_2):
        w_1, x_1, y_1, z_1 = self._val
        w_2, x_2, y_2, z_2 = q_2._val
        w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
        x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
        y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
        z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2

        result = _Quaternion.from_value(np.array((w, x, y, z)))
        return result

    def _multiply_with_vector(self, v):
        q_2 = _Quaternion.from_value(np.append((0.0), v))
        return (self * q_2 * self.get_conjugate())._val[1:]

    def get_conjugate(self):
        w, x, y, z = self._val
        result = _Quaternion.from_value(np.array((w, -x, -y, -z)))
        return result

    def __repr__(self):
        theta, v = self.get_axisangle()
        return f"((%.6f; %.6f, %.6f, %.6f))" % (theta, v[0], v[1], v[2])

    def get_axisangle(self):
        w, v = self._val[0], self._val[1:]
        theta = acos(w) * 2.0

        return theta, _normalize(v)

    def tolist(self):
        return self._val.tolist()

    def vector_norm(self):
        _, v = self.get_axisangle()
        return np.linalg.norm(v)


def visualize_transition(sequence_of_gates,
                         jupyter=False,
                         trace=False,
                         saveas=None):
    """
    Creates animation showing transitions between states of a single
    qubit by applying quantum gates.

    Args:
        sequence_of_gates (list): List of characters that describe sequence
            of gates applied to a single quibit starting from position |0>. Currently
            supports X, Y, Z, S, SDG, H and T gates. e.g. ['X','Y']
        jupyter (bool): Controls whether to display tkinter GUI (when set to False) of the
            animation or return IPython HTML video element of animation to be shown in
            jupyter notebook.
        trace (bool): Controls whether to display tracing vectors - history of 10 past vectors
            at each step of the animation.
        saveas (str): User can choose to save the animation as a video to their filesystem.
            This argument is a string of path with filename and extension (e.g. "movie.mp4" to
            save the video in current working directory).

    Returns:
        IPython.core.display.HTML:
            If arg jupyter is set to True. Otherwise opens tkinter GUI and returns
            after the GUI is closed.

    Raises:
        ImportError: Must have Matplotlib (and/or IPython) installed.
        VisualizationError: Given gate(s) are not supported.

    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Must have Matplotlib installed.")
    if not HAS_IPYTHON and jupyter is True:
        raise ImportError("Must have IPython installed.")

    frames_per_gate = 100
    gates = dict()
    gates['X'] = (_Quaternion.from_axisangle(np.pi / frames_per_gate, [1, 0, 0]), '#000066')
    gates['Y'] = (_Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 1, 0]), '#3333ff')
    gates['Z'] = (_Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 0, 1]), '#6699ff')
    gates['S'] = (_Quaternion.from_axisangle(np.pi / 2 / frames_per_gate, [0, 0, 1]), '#ff3300')
    gates['SDG'] = (_Quaternion.from_axisangle(-np.pi / 2 / frames_per_gate, [0, 0, 1]),
                    '#ff6666')
    gates['H'] = (_Quaternion.from_axisangle(np.pi / frames_per_gate, _normalize([1, 0, 1])),
                  '#9999ff')
    gates['T'] = (_Quaternion.from_axisangle(np.pi / 4 / frames_per_gate, [0, 0, 1]), '#ff33cc')

    if not isinstance(sequence_of_gates, list):
        raise VisualizationError("Input must be a list of gates, e.g. ['X', 'Y']")

    for gate in sequence_of_gates:
        if gate not in gates:
            raise VisualizationError("Given gate(s) are not supported")

    starting_pos = _normalize(np.array([0, 0, 1]))

    fig = plt.figure(figsize=(5, 5))
    _ax = Axes3D(fig)
    _ax.set_xlim(-10, 10)
    _ax.set_ylim(-10, 10)
    sphere = Bloch(axes=_ax)

    class Namespace:
        """Helper class serving as scope container"""
        def __init__(self):
            self.new_vec = []
            self.points = []

    namespace = Namespace()
    namespace.new_vec = starting_pos
    namespace.points = []

    def animate(i):
        sphere.clear()

        gate_counter = i // 100
        namespace.new_vec = gates[sequence_of_gates[gate_counter]][0] * namespace.new_vec
        if i % 10 == 0:
            namespace.points.append(namespace.new_vec)

        sphere.add_vectors(namespace.new_vec)

        if trace:
            if len(namespace.points) < 10:
                sphere.add_vectors(namespace.points)
            else:
                sphere.add_vectors(namespace.points[-10:])

        sphere.vector_color = [gates[sequence_of_gates[gate_counter]][1]]

        annotationvector = [1.4, -0.3, 1.4]
        sphere.add_annotation(annotationvector,
                              sequence_of_gates[gate_counter],
                              color=gates[sequence_of_gates[gate_counter]][1],
                              fontsize=30)

        sphere.make_sphere()
        return _ax

    def init():
        sphere.vector_color = ['r']
        return _ax

    ani = animation.FuncAnimation(fig, animate, range(frames_per_gate * len(sequence_of_gates)),
                                  init_func=init, blit=False, repeat=False, interval=20)

    if saveas:
        ani.save(saveas, fps=30)
    if jupyter:
        return HTML(ani.to_html5_video())

    plt.show()
    plt.close(fig)
    return None
