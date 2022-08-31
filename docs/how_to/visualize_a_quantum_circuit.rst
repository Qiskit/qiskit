===========================
Visualize a quantum circuit
===========================

This guide shows how to get a visual representation of a quantum circuit.

Build the circuit
=================

The first step is to create the circuit.

.. jupyter-execute::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))

Visualize the circuit
=====================

There are three different ways to visualize a circuit. You can use

* The ``print()`` function.
* The :meth:`~qiskit.circuit.QuantumCircuit.draw()` method.
* The :func:`~qiskit.visualization.circuit_drawer()` function.

``print()``
-----------

If you call the ``print()`` function on a :class:`~qiskit.circuit.QuantumCircuit` object, you will get an `ASCII art version <https://en.wikipedia.org/wiki/ASCII_art>`_ of the circuit diagram.

.. jupyter-execute::

    print(qc)

:meth:`~qiskit.circuit.QuantumCircuit.draw()`
---------------------------------------------

You can also call the :meth:`~qiskit.circuit.QuantumCircuit.draw()` method on a :class:`~qiskit.circuit.QuantumCircuit` object to visualize it. If you don't specify any arguments you will get a plain text representation of the circuit.

.. jupyter-execute::

    qc.draw()

However, if you change the ``output`` argument, you can get other different renderings. The available options for this argument are:

* ``'text'``: renders the circuit with ASCII art. It's the default option.
* ``'mpl'``: uses `matplotlib <https://matplotlib.org/>`_ to render the circuit.
* ``'latex'``: uses :math:`\LaTeX` to render the circuit. It requires a full `LaTeX <https://latex.org/forum/>`_ distribution and the package ``pdflatex``.
* ``'latex_source'``: outputs the :math:`\LaTeX` source code that creates the ``'latex'`` rendering of the circuit.

Because this optional or keyword argument is actually the first of this method, one can type ``qc.draw(option)`` instead of ``qc.draw(output=option)``.

.. note::
    By default, the ``draw()`` method returns the rendered image as an object and does not output anything. The exact class returned depends on the output specified: ``'text'`` (the default) returns a ``TextDrawer`` object, ``'mpl'`` returns a ``matplotlib.figure.Figure`` object, and ``'latex'`` returns a ``PIL.Image`` object. Having the return types enables modifying or directly interacting with the rendered output from the drawers. Jupyter notebooks understand these return types and render them for us in this guide, but when running outside of Jupyter, you do not have this feature automatically. However, the ``draw()`` method has optional arguments to display or save the output. When specified, the ``filename`` kwarg takes a path to which it saves the rendered output. Alternatively, if you're using the ``'mpl'`` or ``'latex'`` outputs, you can leverage the ``interactive`` kwarg to open the image in a new window (this will not always work from within a notebook but will be demonstrated anyway).


``'mpl'``
^^^^^^^^^

.. jupyter-execute::

    qc.draw('mpl')



``'latex_source'``
^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    qc.draw('latex_source')


:func:`~qiskit.visualization.circuit_drawer()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to use a self-contained function instead of a :class:`~qiskit.circuit.QuantumCircuit` method to draw your circuit, you can do it with :func:`~qiskit.visualization.circuit_drawer()` from ``qiskit.visualization``. It has the exact same behavior as the :meth:`~qiskit.circuit.QuantumCircuit.draw()` method above, except that it requires the circuit to be included as an argument.

.. note::
    In Qiskit Terra :math:`\leq 0.7`, the default behavior for the ``circuit_drawer()`` function is to use the ``'latex'`` output backend, and in :math:`0.6.x` that includes a fallback to ``'mpl'`` if ``'latex'`` fails for any reason. Starting with release :math:`> 0.7`, the default changes to the ``'text'`` output.


.. jupyter-execute::

    from qiskit.visualization import circuit_drawer

    circuit_drawer(qc, output='mpl')

.. jupyter-execute::

    import qiskit.tools.jupyter
    %qiskit_version_table
    %qiskit_copyright
