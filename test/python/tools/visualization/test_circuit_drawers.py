# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for circuit_drawer."""
import os
import sys
import shutil
import itertools
import copy

import unittest
from unittest.mock import patch, call

from PIL.Image import Image
from PIL import ImageChops
from numpy import pi

import qiskit.tools.visualization._circuit_visualization as _cv
import qiskit.tools.visualization._matplotlib as _mpl

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.visualization import VisualizationError

from qiskit.test import QiskitTestCase, Path


def _this_directory():
    return os.path.dirname(os.path.abspath(__file__))


def _version_to_str():
    return '{}.{}'.format(sys.version_info[0], sys.version_info[1])


def _small_circuit():
    """Creates a simple small circuit consisting of one qubit, one bit and one gate applied.

    Returns:
         QuantumCircuit: the small quantum circuit defined with the function body operations list.

    """
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    circuit = QuantumCircuit(qr, cr)

    circuit.x(qr[0])
    circuit.barrier(qr[0])
    circuit.measure(qr, cr)

    return circuit


def _medium_circuit():
    """Creates a medium-size quantum circuit, consisting of three qubits and most used gates applied
    to them.

    Returns:
         QuantumCircuit: the medium quantum circuit defined with the function body operations list.

    """

    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)

    circuit.x(qr[0])
    circuit.y(qr[1])
    circuit.z(qr[2])

    circuit.h(qr[0])
    circuit.s(qr[1])
    circuit.sdg(qr[2])

    circuit.t(qr[0])
    circuit.tdg(qr[1])
    circuit.iden(qr[2])

    circuit.reset(qr[0])
    circuit.reset(qr[1])
    circuit.reset(qr[2])

    circuit.rx(pi / 8, qr[0])
    circuit.ry(pi / 8, qr[1])
    circuit.rz(pi / 8, qr[2])

    circuit.u1(pi / 8, qr[0])
    circuit.u2(pi / 8, pi / 8, qr[1])
    circuit.u3(pi / 8, pi / 8, pi / 8, qr[2])

    circuit.swap(qr[0], qr[1])

    circuit.cx(qr[0], qr[1])
    circuit.cy(qr[1], qr[2])
    circuit.cz(qr[2], qr[0])
    circuit.ch(qr[0], qr[1])

    circuit.cu1(pi / 8, qr[0], qr[1])
    circuit.cu3(pi / 8, pi / 8, pi / 8, qr[1], qr[2])

    circuit.barrier(qr)

    circuit.measure(qr, cr)

    return circuit


def _large_circuit():
    """Creates a large quantum circuit consisting of nine qubits and most used gates applied to them
    all similarly to the medium quantum circuit.

    Returns:
         QuantumCircuit: the large quantum circuit defined with the function body operations list.

    """
    qr = QuantumRegister(9)
    cr = ClassicalRegister(9)
    circuit = QuantumCircuit(qr, cr)

    for i in range(3):
        zero = 3 * i
        first = 3 * i + 1
        second = 3 * i + 2

        circuit.x(qr[zero])
        circuit.y(qr[first])
        circuit.z(qr[second])

        circuit.h(qr[zero])
        circuit.s(qr[first])
        circuit.sdg(qr[second])

        circuit.t(qr[zero])
        circuit.tdg(qr[first])
        circuit.iden(qr[second])

        circuit.reset(qr[zero])
        circuit.reset(qr[first])
        circuit.reset(qr[second])

        circuit.rx(pi / 8, qr[zero])
        circuit.ry(pi / 8, qr[first])
        circuit.rz(pi / 8, qr[second])

        circuit.u1(pi / 8, qr[zero])
        circuit.u2(pi / 8, pi / 8, qr[first])
        circuit.u3(pi / 8, pi / 8, pi / 8, qr[second])

        circuit.swap(qr[zero], qr[first])

        circuit.cx(qr[zero], qr[first])
        circuit.cy(qr[first], qr[second])
        circuit.cz(qr[second], qr[zero])
        circuit.ch(qr[zero], qr[first])

        circuit.cu1(pi / 8, qr[zero], qr[first])
        circuit.cu3(pi / 8, pi / 8, pi / 8, qr[first], qr[second])

    circuit.barrier(qr)

    circuit.measure(qr, cr)

    return circuit


def _deep_circuit():
    """Creates a very deep quantum circuit with nineteen qubits but poor on operations: there is
    only one gate applied to each qubit.

    Returns:
         QuantumCircuit: the deep quantum circuit defined with the function body operations list.

    """
    qr = QuantumRegister(20)
    cr = ClassicalRegister(20)
    circuit = QuantumCircuit(qr, cr)

    for i in range(10):
        circuit.x(qr[i])
        circuit.h(qr[i])
        circuit.x(qr[i])

    circuit.barrier(qr)

    circuit.measure(qr, cr)

    return circuit


def _reset_registers():
    """Resets counters of created QuantumRegister and ClassicalRegister instances.

    This function is required since references has been obtained such that their QuantumRegisters
    have zero index. Since many circuits are build while testing, indices of correspinding
    registers will be distinct from zero and thus they will lead to test failures unless we reset
    them manually.

    """
    QuantumRegister.instances_counter = itertools.count()
    ClassicalRegister.instances_counter = itertools.count()


def _get_black_pixels(image):
    black_and_white_version = image.convert('1')
    black_pixels = black_and_white_version.histogram()[0]
    return black_pixels


class TestDrawingMethods(QiskitTestCase):
    """This class implements a test case which checks whether outputs of different circuit drawers
    upon drawing different types of QuantumCircuit equal the reference outputs.

    In particular, it provides the tests for:
     1. Small circuit of one qubit;
     2. Medium circuit of three qubits;
     3. Large circuit of nine qubits;
     4. Deep circuit of nineteen qubits.

    Since testing routine is the same for all circuit drawers, each test is a union of subTests,
    each for corresponding circuit drawer (i.e. `test_large_circuit` include four tests: for text,
    latex, latex source and matplotlib drawers).

    Moreover, since tests of different circuit types are almost the same, it is possible to start
    a great test which combines all tests for both different circuit types and circuit drawers.

    """
    def setUp(self):
        # Specify output formats which define a draw method
        self.draw_methods = ('text', 'latex', 'latex_source', 'mpl')

        # Specify extensions of the file obtained during the draw methods invocation
        self.extensions = {'mpl': '.png'}

        # Specify draw methods which produce file or image as output
        self.file_output_methods = ('text', 'latex_source')
        self.image_output_methods = ('latex', 'mpl')

        # Set correspondence between a circuit type and a function to be invoked
        self.circuits = {
            'small': _small_circuit,
            'medium': _medium_circuit,
            'large': _large_circuit,
            'deep': _deep_circuit,
        }

        # Create a temporary folder to store all the outputs produced during testing
        self.tmp_dir = os.path.join(_this_directory(), 'tmp')
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        # This piece of code allows one to easily generate new references just at set up procedure
        # (consequently, all the following test should be successful). Uncomment to use it.
        # for circuit_type in self.circuits:
        #     for draw_method in self.draw_methods:
        #         references_dir = self._get_resource_path(os.path.join(_version_to_str(),
        #                                                               circuit_type),
        #                                                  path=Path.CIRCUIT_DRAWERS_REFERENCES)
        #
        #         references_dir = os.path.join(references_dir)
        #         if not os.path.exists(references_dir):
        #             os.makedirs(references_dir)
        #
        #         reference_output = os.path.join(references_dir, draw_method)
        #
        #         # Make underlying circuit drawer to draw chosen circuit
        #         circuit_drawer(self.circuits[circuit_type](),
        #                        output=draw_method,
        #                        filename=reference_output)
        #
        #         # We have built a QuantumCircuit and counters of registers increased, we have
        #         # reset them such that consequent tests won't fail
        #         _reset_registers()

    def test_small_circuit(self):
        """Tests whether outputs of different circuit drawers upon drawing a small circuit equal
         reference outputs.
        """
        # Specify a type of circuit used in this test
        self.check_circuit_type('small')

    def test_medium_circuit(self):
        """Tests whether outputs of different circuit drawers upon drawing a medium circuit equal
         reference outputs.
        """
        # Specify a type of circuit used in this test
        self.check_circuit_type('medium')

    def test_large_circuit(self):
        """Tests whether outputs of different circuit drawers upon drawing a large circuit equal
         reference outputs.
        """
        # Specify a type of circuit used in this test
        self.check_circuit_type('large')

    def test_deep_circuit(self):
        """Tests whether outputs of different circuit drawers upon drawing a deep circuit equal
         reference outputs.
        """
        # Specify a type of circuit used in this test
        self.check_circuit_type('deep')

    @unittest.skip('A test which runs tests for all circuit types inside is skipped. '
                   'Tests for all circuit types are better to be launched separately.')
    def test_all_circuit_types(self):
        """Adds one more nested loop and this tests whether outputs of different circuit drawers
         upon drawing all types of circuit equal reference outputs.
        """
        for circuit_type in self.circuits:

            # Create a subTest for each type of circuit
            with self.subTest(circuit_type=circuit_type):
                self.check_circuit_type(circuit_type)

    def check_circuit_type(self, circuit_type):
        """Checks whether outputs of different circuit drawers upon drawing a given circuit type
        equal reference outputs.

        Args:
            circuit_type (str): a type of circuit to be checked ('small', 'medium', 'large', or
            'deep')

        Returns:
            bool: True if outputs of all drawers for current circuit type equal reference
            ones up to given precision, False otherwise.
        """
        # Obtain paths to directory where produced and reference outputs are to be stored
        # correspondingly
        test_output_dir, references_dir = self._prepare_dirs('{}'.format(circuit_type))

        for draw_method in self.draw_methods:
            # Create a subTest for each underlying circuit drawer
            with self.subTest('Test of drawing a {} circuit'
                              ' with `{}` output format'.format(circuit_type, draw_method),
                              draw_method=draw_method):
                # Obtain path to files with produced and reference outputs correspondingly
                test_output = os.path.join(test_output_dir, draw_method)
                reference_output = os.path.join(references_dir, draw_method)

                # Make underlying circuit drawer to draw chosen circuit
                circuit_drawer(self.circuits[circuit_type](),
                               output=draw_method,
                               filename=test_output)

                # Check if produced output equals the reference one
                self.assertOutputsAreEqual(draw_method,
                                           test_output + self.extensions.get(draw_method,
                                                                             ''),
                                           reference_output + self.extensions.get(
                                               draw_method, ''))

            # We have built a QuantumCircuit and counters of registers increased, we have
            # reset them such that consequent tests won't fail
            _reset_registers()

    def _prepare_dirs(self, circuit_type):
        # Create a folder to store all the outputs produced during testing of particular circuit
        # type.
        test_output_dir = os.path.join(self.tmp_dir, circuit_type)
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)

        # Simply obtain path to folder with references
        references_dir = self._get_resource_path(os.path.join(_version_to_str(),
                                                              circuit_type),
                                                 path=Path.CIRCUIT_DRAWERS_REFERENCES)
        references_dir = os.path.join(references_dir)

        return test_output_dir, references_dir

    def assertOutputsAreEqual(self, draw_method, test_output, reference_output):
        """Checks if output of some circuit drawer equals the reference one.

        Args:
            draw_method (str): invoked drawing method (`latex`, `latex_source`, `text`
             or `matplotlib`)
            test_output (str): path to circuit drawer output
            reference_output (str): path to reference output

        Returns:
             bool: True if outputs are similar up to given precision, False otherwise.

        """
        if draw_method in self.file_output_methods:
            self.assertFilesAreEqual(test_output, reference_output)

        if draw_method in self.image_output_methods:
            self.assertImagesAreEqual(test_output, reference_output)

    def assertFilesAreEqual(self, current, expected):
        """Checks if both files are the same."""
        self.assertTrue(os.path.exists(current))
        self.assertTrue(os.path.exists(expected))

        with open(current, "r", encoding='cp437') as cur, \
                open(expected, "r", encoding='cp437') as exp:
            self.assertEqual(cur.read(), exp.read(), msg='{} file differs from {}'.format(current,
                                                                                          expected))

    def assertImagesAreEqual(self, current, expected, diff_tolerance=0.001):
        """Checks if both images are similar enough to be considered equal.
        Similarity is controlled with the ```diff_tolerance``` argument."""
        from PIL import Image as im

        current_im = im.open(current)
        expected_im = im.open(expected)

        diff = ImageChops.difference(expected_im, current_im)
        black_pixels = _get_black_pixels(diff)
        total_pixels = diff.size[0] * diff.size[1]
        similarity_ratio = black_pixels / total_pixels
        self.assertTrue(
            1 - similarity_ratio < diff_tolerance,
            msg='The image {} differs from {} by more than a {}%'
            .format(current, expected, diff_tolerance * 100))

    def tearDown(self):
        # Mercilessly delete a temporary folder
        shutil.rmtree(self.tmp_dir)

        super(TestDrawingMethods, self).tearDown()


class TestCircuitDrawer(QiskitTestCase):
    """This class implements a test case to check how circuit_drawer behaves upon different
    values of `output` parameter.

    In particular, it tests handling of the following cases:
    1) Correct value of `output` provided;
    2) Wrong value of `output` provided;
    3) No value of `output` provided.

    Also, this test case checks how `interactive=True` is handled.
    """

    def setUp(self):
        # Specify correspondence between output format and called circuit drawer function
        self.draw_methods = {
            'text': '_text_circuit_drawer',
            'latex': '_latex_circuit_drawer',
            'latex_source': '_generate_latex_source',
            'mpl': '_matplotlib_circuit_drawer'
        }

        # Specify methods which imply testing of the interactive mode
        self.interactive_draw_methods = ('latex', 'mpl')

        # Specify which arguments shall be passed to the mocked call of corresponding draw method
        self.calls = {
            'text': {
                'filename': None,
                'line_length': None,
                'reversebits': False,
                'plotbarriers': True,
            },
            'latex': {
                'scale': 0.7,
                'filename': None,
                'style': None,
                'plot_barriers': True,
                'reverse_bits': False
            },
            'latex_source': {
                'scale': 0.7,
                'filename': None,
                'style': None,
                'plot_barriers': True,
                'reverse_bits': False
            },
            'mpl': {
                'scale': 0.7,
                'filename': None,
                'style': None,
                'plot_barriers': True,
                'reverse_bits': False
            },
        }

    def test_correct_output_provided(self):
        """Tests how correctly specified output circuit drawer is called.

        Since testing routine is the same for all four circuit drawers,
        test is split into subtests, one for corresponding circuit drawer.

        """
        # Create a subTest for current circuit drawer
        for draw_method in self.draw_methods:
            with self.subTest('Test calling of the {} draw method'.format(draw_method),
                              draw_method=draw_method):

                # Patch function corresponding to the current circuit drawer such that
                # it does nothing
                with patch.object(_cv, self.draw_methods[draw_method], return_value=None)\
                        as mock_draw_method:

                    # Check that corresponding function was called once with the correct arguments
                    circuit_drawer(None, output=draw_method)
                    mock_draw_method.assert_called_once_with(None, **self.calls[draw_method])

    def test_wrong_output_provided(self):
        """Tests correct exceptioning of wrong output format """
        with self.assertRaises(VisualizationError):
            circuit_drawer(None, output='wrong_output')

    @unittest.skip('Test for deprecated fallback is skipped ')
    def test_no_output_provided(self):
        """Tests how `circuit_drawer` falls back when no `output` is given

        First, it creates a subTest to check how `circuit_drawer` falls back
        to LaTeX circuit drawer.

        Next, it creates a subTest to test fallback to matplotlib circuit drawer.
        Here, it checks two cases: `HAS_MATPLOTLIB=True` and `HAS_MATPLOTLIB=False`.

        """
        # Create a subTest for fallback to latex_circuit_drawer
        with self.subTest('Test fallback to latex_circuit_drawer'):

            # Patch _latex_circuit_drawer such that it does nothing
            with patch.object(_cv, self.draw_methods['latex'], return_value=None)\
                    as mock_latex_circuit_drawer:

                # Check that _latex_circuit_drawer was called once with the correct arguments
                circuit_drawer(None)
                mock_latex_circuit_drawer.assert_called_once_with(call(None,
                                                                       scale=0.7,
                                                                       filename=None,
                                                                       style=None,
                                                                       output='text',
                                                                       interactive=False,
                                                                       line_length=None,
                                                                       plot_barriers=True,
                                                                       reverse_bits=False))

        # Create a subTest for fallback to matplotlib_circuit_drawer
        with self.subTest('Test fallback to matplotlib_circuit_drawer'):

            # Patch _latex_circuit_drawer such that it raises FileNotFoundError
            with patch.object(_cv, self.draw_methods['latex'], side_effect=FileNotFoundError) as _:

                # Patch HAS_MATPLOTLIB attribute of _matplotlib to True
                with patch.object(_mpl, 'HAS_MATPLOTLIB', return_value=True):

                    # Patch _matplotlib_circuit_drawer such that it does nothing
                    with patch.object(_cv, self.draw_methods['mpl'])\
                            as mock_mpl_circuit_drawer:

                        # Check that _matplotlib_circuit_drawer was called once with the correct
                        # arguments
                        circuit_drawer(None)
                        mock_mpl_circuit_drawer.assert_called_once_with(call(None,
                                                                             scale=0.7,
                                                                             filename=None,
                                                                             style=None,
                                                                             output='text',
                                                                             interactive=False,
                                                                             line_length=None,
                                                                             plot_barriers=True,
                                                                             reverse_bits=False))

                # Patch HAS_MATPLOTLIB attribute of _matplotlib to False
                with patch.object(_mpl, 'HAS_MATPLOTLIB', new=False):

                    # Check that circuit_drawer raises correct exception
                    with self.assertRaises(ImportError):
                        circuit_drawer(None)

    def test_interactive(self):
        """Tests that `interactive=True` makes Image to be shown.

        Since testing routine is the same for both circuit drawers
        that support interactivity, test is split into subtests,
        one for corresponding circuit drawer.
        """
        # Create a subTest for current circuit drawer
        for draw_method in self.interactive_draw_methods:
            with self.subTest('Test interactive regime for {} output'.format(draw_method),
                              draw_method=draw_method):

                # Patch corresponding circuit_drawer such that it returns an instance of Image
                with patch.object(_cv, self.draw_methods[draw_method], return_value=Image()) as _:

                    # Patch show attribute of Image such that it does nothing
                    with patch.object(Image, 'show', return_value=None) as mock_show:

                        # Check that show was called once with the correct arguments
                        circuit_drawer(None, output=draw_method, interactive=True)
                        mock_show.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
