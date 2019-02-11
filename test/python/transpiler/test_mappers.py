# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Meta tests for mappers.

The test checks the output of the swapper to a ground truth DAG (one for each
test/swapper) saved in as a pickle (in `test/python/pickles/`). If they need
to be regenerated, the DAG candidate is compiled and run in a simulator and
the count is checked before being saved. This happens with (in the root
directory):

> python -m  test.python.transpiler.test_mappers

To make a new swapper pass throw all the common tests, create a new class inside the file
`path/to/test_mappers.py` that:
    * the class name should start with `Tests...`.
    * inheriting from ``CommonTestCases, QiskitTestCase``
    * overwrite the required attribute ``pass_class``

For example::

    class TestsSomeSwap(CommonTestCases, QiskitTestCase):
        pass_class = SomeSwap           # The pass class
        additional_args = {'seed': 42}  # In case SomeSwap.__init__ requires
                                        # additional arguments

To **add a test for all the swappers**: add a new method ``test_foo``to the
``CommonTestCases`` class:
    * defining the following required ``self`` attributes: ``self.count``,
      ``self.shots``, ``self.delta``. They are required for the regeneration of the
      ground truth.
    * use the ``self.assertResult`` assertion for comparing for regeneration of the
      ground truth.
    * explicitly set a unique ``name`` of the ``QuantumCircuit``, as it it used
      for the name of the pickle file of the ground truth.

For example::

    def test_a_common_test(self):
        self.count = {'000': 512, '110': 512}  # The expected count for this circuit
        self.shots = 1024                      # Shots to run in the backend.
        self.delta = 5                         # This is delta for the AlmostEqual during
                                               # the count check
        coupling_map = [[0, 1], [0, 2]]        # The coupling map for this specific test

        qr = QuantumRegister(3, 'q')               #
        cr = ClassicalRegister(3, 'c')             # Set the circuit to test
        circuit = QuantumCircuit(qr, cr,           # and don't forget to put a name
                                 name='some_name') # (it will be used to save the pickle
        circuit.h(qr[1])                           #
        circuit.cx(qr[1], qr[2])                   #
        circuit.measure(qr, cr)                    #

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                           pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, circuit.name)
```
"""

# pylint: disable=redefined-builtin,attribute-defined-outside-init

import unittest
import pickle
import sys

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, BasicAer, compile
from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, StochasticSwap
from qiskit.mapper import CouplingMap, Layout

from qiskit.test import QiskitTestCase


class CommonUtilitiesMixin:
    """Utilities for meta testing.

    Subclasses should redefine the ``pass_class`` argument, with a Swap Mapper
    class.

    Note: This class assumes that the subclass is also inheriting from
    ``QiskitTestCase``, and it uses ``QiskitTestCase`` methods directly.
    """

    regenerate_expected = False
    seed = 42
    additional_args = {}
    pass_class = None

    def create_passmanager(self, coupling_map, initial_layout=None):
        """Returns a PassManager using self.pass_class(coupling_map, initial_layout)"""
        passmanager = PassManager(
            self.pass_class(CouplingMap(coupling_map),  # pylint: disable=not-callable
                            **self.additional_args))
        if initial_layout:
            passmanager.property_set['layout'] = Layout(initial_layout)
        return passmanager

    def create_backend(self):
        """Returns a Backend."""
        return BasicAer.get_backend('qasm_simulator')

    def generate_expected(self, transpiled_result, filename):
        """Generates the expected result into a file.

        Checks if transpiled_result matches self.counts by running in a backend
        (self.create_backend()). That's saved in a pickle in filename.

        Args:
            transpiled_result (DAGCircuit): The DAGCircuit to compile and run.
            filename (string): Where the pickle is saved.
        """
        sim_backend = self.create_backend()
        qobj = compile(transpiled_result, sim_backend, seed=self.seed, shots=self.shots)
        job = sim_backend.run(qobj)
        self.assertDictAlmostEqual(self.counts, job.result().get_counts(), delta=self.delta)

        with open(filename, "wb") as output_file:
            pickle.dump(transpiled_result, output_file)

    def assertResult(self, result, testname):
        """Fetches the pickle in testname file and compares it with result."""
        filename = self._get_resource_path('pickles/%s_%s.pickle' % (type(self).__name__, testname))

        if self.regenerate_expected:
            # Run result in backend to test that is valid.
            self.generate_expected(result, filename)

        with open(filename, "rb") as input_file:
            expected = pickle.load(input_file)

        self.assertEqual(result, expected)


class CommonTestCases(CommonUtilitiesMixin):
    """Tests that are run in several mappers.

    The tests here will be run in several mappers. When adding a test, please
    ensure that the test:

    * defines ``self.count``, ``self.shots``, ``self.delta``.
    * uses the ``self.assertResult`` assertion for comparing for regeneration of
      the ground truth.
    * explicitly sets a unique ``name`` of the ``QuantumCircuit``.

    See also ``CommonUtilitiesMixin`` and the module docstring.
    """

    def test_a_cx_to_map(self):
        """A single CX needs to be remapped.

         q0:----------m-----
                      |
         q1:-[H]-(+)--|-m---
                  |   | |
         q2:------.---|-|-m-
                      | | |
         c0:----------.-|-|-
         c1:------------.-|-
         c2:--------------.-

         CouplingMap map: [1]<-[0]->[2]

        expected count: '000': 50%
                        '110': 50%
        """
        self.counts = {'000': 512, '110': 512}
        self.shots = 1024
        self.delta = 5
        coupling_map = [[0, 1], [0, 2]]

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr, name='a_cx_to_map')
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr, cr)

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                           pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, circuit.name)

    def test_initial_layout(self):
        """Using a non-trivial initial_layout.

         q3:----------------m--
         q0:----------m-----|--
                      |     |
         q1:-[H]-(+)--|-m---|--
                  |   | |   |
         q2:------.---|-|-m-|--
                      | | | |
         c0:----------.-|-|-|--
         c1:------------.-|-|--
         c2:--------------.-|--
         c3:----------------.--
         CouplingMap map: [1]<-[0]->[2]->[3]

        expected count: '000': 50%
                        '110': 50%
        """
        self.counts = {'0000': 512, '0110': 512}
        self.shots = 1024
        self.delta = 5
        coupling_map = [[0, 1], [0, 2], [2, 3]]

        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr, name='initial_layout')
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr, cr)

        layout = [qr[3], qr[0], qr[1], qr[2]]

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                           pass_manager=self.create_passmanager(coupling_map, layout))
        self.assertResult(result, circuit.name)

    def test_handle_measurement(self):
        """Handle measurement correctly.

         q0:--.-----(+)-m-------
              |      |  |
         q1:-(+)-(+)-|--|-m-----
                  |  |  | |
         q2:------|--|--|-|-m---
                  |  |  | | |
         q3:-[H]--.--.--|-|-|-m-
                        | | | |
         c0:------------.-|-|-|-
         c1:--------------.-|-|-
         c2:----------------.-|-
         c3:------------------.-

         CouplingMap map: [0]->[1]->[2]->[3]

        expected count: '0000': 50%
                        '1011': 50%
        """
        self.counts = {'1011': 512, '0000': 512}
        self.shots = 1024
        self.delta = 5
        coupling_map = [[0, 1], [1, 2], [2, 3]]

        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr, name='handle_measurement')
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[3], qr[1])
        circuit.cx(qr[3], qr[0])
        circuit.measure(qr, cr)

        result = transpile(circuit, self.create_backend(), coupling_map=coupling_map,
                           pass_manager=self.create_passmanager(coupling_map))
        self.assertResult(result, circuit.name)


class TestsBasicSwap(CommonTestCases, QiskitTestCase):
    """Test CommonTestCases using BasicSwap."""
    pass_class = BasicSwap


class TestsLookaheadSwap(CommonTestCases, QiskitTestCase):
    """Test CommonTestCases using LookaheadSwap."""
    pass_class = LookaheadSwap


class TestsStochasticSwap(CommonTestCases, QiskitTestCase):
    """Test CommonTestCases using StochasticSwap."""
    pass_class = StochasticSwap
    additional_args = {'seed': 0}


if __name__ == '__main__':
    if len(sys.argv) >=2 and sys.argv[1] == 'regenerate':
        CommonUtilitiesMixin.regenerate_expected = True
        del sys.argv[1]
    unittest.main()
