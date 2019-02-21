


Contributing to Qiskit
======================

Qiskit is an open-source project committed to bringing quantum computing to people of all
backgrounds. This page describes how you can join the Qiskit community in this goal.



Where Things Are
----------------

The code for Qiskit is located in the `Qiskit GitHub organization <https://github.com/Qiskit>`_, where you can find the individual projects that make up Qiskit, including

* `Qiskit Terra <https://github.com/Qiskit/qiskit-terra>`_
* `Qiskit Aer <https://github.com/Qiskit/qiskit-aer>`_
* `Qiskit Aqua <https://github.com/Qiskit/qiskit-aqua>`_
* `Qiskit Chemistry <https://github.com/Qiskit/qiskit-chemistry>`_
* `Qiskit IBMQ Provider <https://github.com/Qiskit/qiskit-ibmq-provider>`_
* `Qiskit Tutorials <https://github.com/Qiskit/qiskit-tutorials>`_
* `Qiskit Documentation <https://github.com/Qiskit/qiskit/tree/master/docs>`_



Getting Started
---------------

Learn how members of the Qiskit community

* `relate to one another <https://github.com/Qiskit/qiskit/blob/master/.github/CODE_OF_CONDUCT.md>`_
* `discuss ideas <https://qiskit.slack.com/>`_
* `get help when we're stuck <https://quantumcomputing.stackexchange.com/questions/tagged/qiskit>`_
* `stay informed of news in the community <https://medium.com/qiskit>`_
* `keep a consistent style <https://www.python.org/dev/peps/pep-0008>`_
* `work together on GitHub <https://github.com/Qiskit/qiskit/blob/master/.github/CONTRIBUTING.md>`_
* :ref:`build Qiskit packages from source <install_install_from_source_label>`



Writing Documentation
---------------------

Qiskit documentation is shaped by the `docs as code <https://www.writethedocs.org/guide/docs-as-code/>`_ philosophy.

The `published documentation <https://qiskit.org/documentation/index.html>`_ is built from the master branch of `Qiskit/qiskit/docs <https://github.com/Qiskit/qiskit/tree/master/docs>`_ using `Sphinx <http://www.sphinx-doc.org/en/master/>`_.

The Python API reference documentation is automatically generated from comments in the code by navigating to your local clone of `Qiskit/qiskit <https://github.com/Qiskit/qiskit>`_ and running the following command in a terminal window:

.. code:: sh

  make doc



Creating a Custom Provider
--------------------------

This example discusses how to approach the design and implementation of
a Qiskit provider, described in `Advanced Use of IBM Q
Devices <https://qiskit.org/documentation/advanced_use_of_ibm_q_devices.html>`__.
The objective of the provider in this example is to simulate circuits
made up entirely of Hadamard gates terminating with measurement of all
qubits.



Designing the Backend Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To design a provider that simulates Hadamard gates, first determine the
elements of data and operations that form your *backend*, a simulator
(in this example) or real quantum computer responsible for running
circuits and returning results. A backend has at least these properties:

-  ``name``
-  ``configuration``
-  ``status``
-  ``provider``

You must be able to perform certain operations on a backend:

-  Translate a circuit, described by a ``qobj`` (a quantum object) into
   a form expected by your simulator
-  Run your simulator
-  Return a ``BaseJob`` object that contains the results of your
   simulator

The ``run()`` method implements these operations.



**Implementing the Backend Class**

.. code:: python

    from qiskit.providers import BaseBackend
    from qiskit.providers.models import BackendConfiguration
    from qiskit import qobj as qiskit_qobj
    from qiskit.result import Result

    # Inherit from qiskit.providers.BaseBackend
    class HadamardSimulator(BaseBackend):

        def __init__(self, provider=None):

            configuration = {
                'backend_name': 'hadamard_simulator',
                'backend_version': '0.1.0',
                'url': 'http://www.i_love_hadamard.com',
                'simulator': True,
                'local': True,
                'description': 'Simulates only Hadamard gates',
                # basis_gates must contain at least two gates
                'basis_gates': ['h', 'x'],
                'memory': True,
                'n_qubits': 30,
                'conditional': False,
                'max_shots': 100000,
                'open_pulse': False,
                'gates': [
                    {
                        'name': 'TODO',
                        'parameters': [],
                        'qasm_def': 'TODO'
                    }
                ]
            }

            # The provider will be explained in a section below
            super().__init__(
                configuration=BackendConfiguration.from_dict(
                    configuration),
                provider=provider)


        def run(self, qobj):

            # The job object will be explained in a section below
            hadamard_job = HadamardJob(None)

            # Simulate each circuit described by the qobj
            experiment_results = []
            for circuit_index, circuit \
                in enumerate(qobj.experiments):

                number_of_qubits = circuit.config.n_qubits
                shots = qobj.config.shots

                # Need to ensure that the circuit described by qobj
                # only has gates our simulator can handle.
                # We take this for granted here.

                list_of_qubits = []
                for operation in circuit.instructions:
                    if operation.name == 'h':
                        list_of_qubits.append(operation.qubits[0])

                # Need to verify that all the qubits are measured,
                # and to different classical registers.
                # We take this for granted here.

                # Run the Hadamard simulator, discussed below
                counts = run_hadamard_simulator(number_of_qubits,
                  list_of_qubits, shots)

                # Format results for printing
                formatted_counts = {}
                for i in range(2**number_of_qubits):
                    if counts[i] != 0:
                        formatted_counts[hex(i)] = counts[i]

                experiment_results.append({
                    'name': circuit.header.name,
                    'success': True,
                    'shots': shots,
                    'data': {'counts': formatted_counts},
                    'header': circuit.header.as_dict()
                })

            # Return the simulation results in the job object
            hadamard_job._result = Result.from_dict({
                'results': experiment_results,
                'backend_name': 'hadamard_simulator',
                'backend_version': '0.1.0',
                'qobj_id': '0',
                'job_id': '0',
                'success': True
            })

            return hadamard_job



Designing the Job Class
^^^^^^^^^^^^^^^^^^^^^^^

Job instances can be thought of as the “ticket” for a submitted job.
They find out the execution’s state at a given point in time (for
example, if the job is queued, running, or has failed) and also allow
control over the job.

The ``HadamardJob`` class stores information about itself and the
simulation results in the following properties:

-  ``job_id``
-  ``backend`` - The backend the job was run on

The ``HadamardJob`` class performs the following operations:

-  ``result`` - get the result of a ``run`` on the backend
-  ``status``
-  ``cancel``
-  ``submit``

In this example, we will only implement a method for the `result` operation.

**Implementing the Job Class**

Define a simple implementation of a job class that can merely return the
simulation results.

.. code:: python

    from qiskit.providers import BaseJob

    # Inherits from qiskit.providers.BaseJob
    class HadamardJob(BaseJob):
        def __init__(self, backend):
            super().__init__(backend, 1)

        def result(self):
            return self._result

        def cancel(self):
            pass

        def status(self):
            pass

        def submit(self):
            pass



Designing the Provider Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A provider is an entity that gives access to a group of different
backends. A provider must be able to

-  return all backends known to it
-  return a backend queried by name

The ``HadamardProvider`` class implements two methods:

-  ``backends`` - Method that lists all known backends
-  ``get_backend`` - Method that returns backends by name.



**Implementing the Provider Class**

.. code:: python

    from qiskit.providers import BaseProvider
    from qiskit.providers.providerutils import filter_backends

    # Inherits from qiskit.providers.BaseProvider
    class HadamardProvider(BaseProvider):

        def __init__(self, *args, **kwargs):
            super().__init__(args, kwargs)

            # Populate the list of Hadamard backends
            self._backends = [HadamardSimulator(provider=self)]

        def get_backend(self, name=None, **kwargs):
            return super().get_backend(name=name, **kwargs)

        def backends(self, name=None, filters=None, **kwargs):
            # pylint: disable=arguments-differ
            backends = self._backends
            if name:
                backends = [backend for backend in backends
                            if backend.name() == name]

            return filter_backends(
                backends, filters=filters, **kwargs)

        def __str__(self):
            return 'HadamardProvider'



Implementing a Custom Simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulator accepts only a single quantum circuit, where all the gates
are Hadamard gates, and all qubits are measured at the end. The input
format is a list of qubits on whom Hadamard gates are applied. The
simulator returns the counts of each basis state, in the form of a list,
where the basis states are assumed to be ordered lexicographically.

.. code:: python

    def run_hadamard_simulator(number_of_qubits, list_of_qubits, shots):

        # For each qubit, store whether it is manipulated
        # by an odd number of Hadamard gates
        # Example: for run_hadamard_simulator(5, [3, 1, 3, 4], 100)
        # we obtain hadamard_list:
        # [0, 1, 0, 0, 1]
        # because qubits 1 and 4 have
        # an odd number of Hadamard gates.
        hadamard_list = [0]*number_of_qubits
        for qubit in list_of_qubits:
            hadamard_list[qubit] = (1 + hadamard_list[qubit])%2

        # Calculate the result for each basis state
        result = [0]*(2**number_of_qubits)
        for i in range(2**number_of_qubits):
            # Example: when i is 2,
            # the basis_state is 01000
            basis_state = \
                '{0:b}'.format(i).zfill(number_of_qubits)[::-1]

            for qubit in range(number_of_qubits):
                if (hadamard_list[qubit] == 0
                    and basis_state[qubit] == '1'):
                    result[i] = 0
                    break
                if hadamard_list[qubit] == 1:
                    result[i] += int(
                        shots/(2**(1 + hadamard_list.count(1))))

        return result



Using Custom Providers
^^^^^^^^^^^^^^^^^^^^^^

The following code runs two simulators on the same quantum circuit. The
simulators are accessed by their providers.

.. code:: python

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
    from qiskit.transpiler import PassManager

    # Create a circuit with just Hadamards and measurements
    qreg = QuantumRegister(4)
    creg = ClassicalRegister(4)
    qc = QuantumCircuit(qreg, creg)
    qc.h(qreg[3])
    qc.h(qreg[1])
    qc.h(qreg[3])
    qc.h(qreg[2])
    qc.measure(qreg, creg)

    # Use the custom provider to simulate the circuit
    hadamard_provider = HadamardProvider()

    hadamard_job = execute(qc, hadamard_provider.get_backend('hadamard_simulator'), pass_manager=PassManager(), shots=1024)

    hadamard_result = hadamard_job.result()

    # Use an Aer provider to compare and contrast
    aer_job = execute(qc, Aer.get_backend('qasm_simulator'),
        pass_manager=PassManager(), shots=1024)

    aer_result = aer_job.result()

    # Print the results of both providers
    print('Hadamard simulator:')
    print(hadamard_result.get_counts(qc))
    print('Aer simulator:')
    print(aer_result.get_counts(qc))


.. parsed-literal::

    Hadamard simulator:
    {'0100': 256, '0000': 256, '0010': 256, '0110': 256}
    Aer simulator:
    {'0100': 266, '0000': 252, '0010': 233, '0110': 273}
