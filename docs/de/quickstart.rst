Erste Schritte
==============

Den Startpunkt zum Schreiben von Code stellt der
:class:`~qiskit.QuantumCircuit` dar. Bei einem Circuit (oder Score falls
Sie bereits Erfahrungen mit der Quantum Experience haben) handelt es sich um
eine Sammlung von :class:`~qiskit.ClassicalRegister` Objekten,
:class:`~qiskit.QuantumRegister` Objekten und
:mod:`gates <qiskit.extensions.standard>`. Durch die
:ref:`Top-Level Funktion <qiskit_top_level_functions>` können Circuits zu
einem entfernten Quantum Device oder einen lokalen Simulator Backend gesendet
und die Resultate zur weiteren Auswerten gesammelt werden.

Um einen Circuit zur erzeugen und auf einem Simulator, der in Qiskit
enthalten ist, auszuführen, reicht folgender Code:

.. code-block:: python

    # Import the Qiskit SDK
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import available_backends, execute

    # Create a Quantum Register with 2 qubits.
    q = QuantumRegister(2)
    # Create a Classical Register with 2 bits.
    c = ClassicalRegister(2)
    # Create a Quantum Circuit
    qc = QuantumCircuit(q, c)

    # Add a H gate on qubit 0, putting this qubit in superposition.
    qc.h(q[0])
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
    # the qubits in a Bell state.
    qc.cx(q[0], q[1])
    # Add a Measure gate to see the state.
    qc.measure(q, c)

    # See a list of available local simulators
    print("Local backends: ", available_backends({'local': True}))

    # Compile and run the Quantum circuit on a simulator backend
    job_sim = execute(qc, "local_qasm_simulator")
    sim_result = job_sim.result()

    # Show the results
    print("simulation: ", sim_result)
    print(sim_result.get_counts(qc))

Die :func:`~qiskit.Result.get_counts` Methode gibt ein Dictionary von
``state:counts`` Paaren aus.

.. code-block:: python

    {'00': 531, '11': 493}

Quantum Chips
-------------

Man kann QASM Circuits auf echter Hardware ausführen, indem man die IBM Q
Experience (QX) Cloud Platform verwendet. Zur Zeit können Sie aus folgenden
Chips wählen:

-   ``ibmqx4``: `5-Qubit Backend <https://ibm.biz/qiskit-ibmqx4>`_

-   ``ibmqx5``: `16-Qubit Backend <https://ibm.biz/qiskit-ibmqx5>`_

Für weitere Details zu den Chips und Echtzeit Informationen zur Verfügbarkeit
besuchen Sie bitte die `IBM Q Experience Backend Information <https://github
.com/Qiskit/ibmqx-backend-information>`_ Seite sowie die `IBM Q
Experience Geräte <https://quantumexperience.ng.bluemix.net/qx/devices>`_
Seite.

.. include:: example_real_backend.rst

Projekt Organisation
--------------------

Python Beispielprogramme finden Sie im Ordner *examples* und Testskripte im
Ordner *test*. Der *qiskit* Ordner beinhaltet das Hauptmodul
des SDK.
