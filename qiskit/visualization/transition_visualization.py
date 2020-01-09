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
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
    from qiskit.visualization.bloch import Bloch
    from qiskit.visualization.exceptions import VisualizationError
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from IPython.display import HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


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
        """Create quaternion from axis"""
        v = _normalize(v)

        new_quaternion = _Quaternion()
        new_quaternion._axisangle_to_q(theta, v)
        return new_quaternion

    @staticmethod
    def from_value(value):
        """Create quaternion from vector"""
        new_quaternion = _Quaternion()
        new_quaternion._val = value
        return new_quaternion

    def _axisangle_to_q(self, theta, v):
        """Convert axis and angle to quaternion"""
        x = v[0]
        y = v[1]
        z = v[2]

        w = cos(theta/2.)
        x = x * sin(theta/2.)
        y = y * sin(theta/2.)
        z = z * sin(theta/2.)

        self._val = np.array([w, x, y, z])

    def __mul__(self, b):
        """Multiplication of quaternion with quaternion or vector"""

        if isinstance(b, _Quaternion):
            return self._multiply_with_quaternion(b)
        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) != 3:
                raise Exception("Input vector has invalid length {0}".format(len(b)))
            return self._multiply_with_vector(b)
        else:
            raise Exception("Multiplication with unknown type {0}".format(type(b)))

    def _multiply_with_quaternion(self, q_2):
        """Multiplication of quaternion with quaternion"""
        w_1, x_1, y_1, z_1 = self._val
        w_2, x_2, y_2, z_2 = q_2._val
        w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
        x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
        y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
        z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2

        result = _Quaternion.from_value(np.array((w, x, y, z)))
        return result

    def _multiply_with_vector(self, v):
        """Multiplication of quaternion with vector"""
        q_2 = _Quaternion.from_value(np.append((0.0), v))
        return (self * q_2 * self.get_conjugate())._val[1:]

    def get_conjugate(self):
        """Conjugation of quaternion"""
        w, x, y, z = self._val
        result = _Quaternion.from_value(np.array((w, -x, -y, -z)))
        return result

    def __repr__(self):
        theta, v = self.get_axisangle()
        return "(({0}; {1}, {2}, {3}))".format(theta, v[0], v[1], v[2])

    def get_axisangle(self):
        """Returns angle and vector of quaternion"""
        w, v = self._val[0], self._val[1:]
        theta = acos(w) * 2.0

        return theta, _normalize(v)

    def tolist(self):
        """Converts quaternion to a list"""
        return self._val.tolist()

    def vector_norm(self):
        """Calculates norm of quaternion"""
        _, v = self.get_axisangle()
        return np.linalg.norm(v)


def visualize_transition(circuit,
                         jupyter=False,
                         trace=False,
                         saveas=None,
                         fpg=100,
                         spg=2):
    """
    Creates animation showing transitions between states of a single
    qubit by applying quantum gates.

    Args:
        circuit (QuantumCircuit): Qiskit single-qubit QuantumCircuit. Gates supported are
            h,x, y, z, rx, ry, rz, s, sdg, t, tdg and u1.
        jupyter (bool): Controls whether to display tkinter GUI (when set to False) of the
            animation or return IPython HTML video element of animation to be shown in
            jupyter notebook.
        trace (bool): Controls whether to display tracing vectors - history of 10 past vectors
            at each step of the animation.
        saveas (str): User can choose to save the animation as a video to their filesystem.
            This argument is a string of path with filename and extension (e.g. "movie.mp4" to
            save the video in current working directory).
        fpg (int): Frames per gate. Finer control over animation smoothness and computiational
            needs to render the animation. Works well for tkinter GUI as it is, for jupyter GUI
            it might be preferable to choose fpg between 5-30.
        spg (int): Seconds per gate. How many seconds should animation of individual gate
            transitions take.

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
    if len(circuit.qubits) != 1:
        raise VisualizationError("Only one quibit circuits are supported")

    frames_per_gate = fpg
    time_between_frames = (spg*1000)/fpg

    # quaternions of gates which don't take parameters
    gates = dict()
    gates['x'] = ('x', _Quaternion.from_axisangle(np.pi / frames_per_gate, [1, 0, 0]), '#1abc9c')
    gates['y'] = ('y', _Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 1, 0]), '#2ecc71')
    gates['z'] = ('z', _Quaternion.from_axisangle(np.pi / frames_per_gate, [0, 0, 1]), '#3498db')
    gates['s'] = ('s', _Quaternion.from_axisangle(np.pi / 2 / frames_per_gate,
                                                  [0, 0, 1]), '#9b59b6')
    gates['sdg'] = ('sdg', _Quaternion.from_axisangle(-np.pi / 2 / frames_per_gate, [0, 0, 1]),
                    '#8e44ad')
    gates['h'] = ('h', _Quaternion.from_axisangle(np.pi / frames_per_gate, _normalize([1, 0, 1])),
                  '#34495e')
    gates['t'] = ('t', _Quaternion.from_axisangle(np.pi / 4 / frames_per_gate, [0, 0, 1]),
                  '#e74c3c')
    gates['tdg'] = ('tdg', _Quaternion.from_axisangle(-np.pi / 4 / frames_per_gate, [0, 0, 1]),
                    '#c0392b')

    implemented_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 's', 'sdg', 't', 'tdg', 'u1']
    simple_gates = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg']
    list_of_circuit_gates = []

    for gate in circuit._data:
        if gate[0].name not in implemented_gates:
            raise VisualizationError("Gate {0} is not supported".format(gate[0].name))
        if gate[0].name in simple_gates:
            list_of_circuit_gates.append(gates[gate[0].name])
        else:
            theta = gate[0].params[0]
            rad = np.deg2rad(theta)
            if gate[0].name == 'rx':
                quaternion = _Quaternion.from_axisangle(rad / frames_per_gate, [1, 0, 0])
                list_of_circuit_gates.append(('rx:'+str(theta), quaternion, '#16a085'))
            elif gate[0].name == 'ry':
                quaternion = _Quaternion.from_axisangle(rad / frames_per_gate, [0, 1, 0])
                list_of_circuit_gates.append(('ry:'+str(theta), quaternion, '#27ae60'))
            elif gate[0].name == 'rz':
                quaternion = _Quaternion.from_axisangle(rad / frames_per_gate, [0, 0, 1])
                list_of_circuit_gates.append(('rz:'+str(theta), quaternion, '#2980b9'))
            elif gate[0].name == 'u1':
                quaternion = _Quaternion.from_axisangle(rad / frames_per_gate, [0, 0, 1])
                list_of_circuit_gates.append(('u1:'+str(theta), quaternion, '#f1c40f'))

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
    namespace.points.append(starting_pos)

    def animate(i):
        sphere.clear()

        # starts with default vector [0,0,1]
        if i == 0:
            sphere.add_vectors(namespace.new_vec)
            sphere.make_sphere()
            return _ax

        gate_counter = (i-1) // frames_per_gate
        namespace.new_vec = list_of_circuit_gates[gate_counter][1] * namespace.new_vec
        if (i-1) % 10 == 0:
            namespace.points.append(namespace.new_vec)

        sphere.add_vectors(namespace.new_vec)

        if trace:
            if len(namespace.points) < 10:
                sphere.add_vectors(namespace.points)
            else:
                sphere.add_vectors(namespace.points[-10:])

        sphere.vector_color = [list_of_circuit_gates[2][2]]

        annotation_text = list_of_circuit_gates[gate_counter][0]
        annotationvector = [1.4, -0.45, 1.7]
        sphere.add_annotation(annotationvector,
                              annotation_text,
                              color=list_of_circuit_gates[gate_counter][2],
                              fontsize=30,
                              horizontalalignment='left')

        sphere.make_sphere()
        return _ax

    def init():
        sphere.vector_color = ['r']
        return _ax

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  range(frames_per_gate * len(list_of_circuit_gates)+1),
                                  init_func=init,
                                  blit=False,
                                  repeat=False,
                                  interval=time_between_frames)

    if saveas:
        ani.save(saveas, fps=30)
    if jupyter:
        # This is necessary to overcome matplotlib memory limit
        matplotlib.rcParams['animation.embed_limit'] = 50
        return HTML(ani.to_jshtml())
    plt.show()
    plt.close(fig)
    return None
