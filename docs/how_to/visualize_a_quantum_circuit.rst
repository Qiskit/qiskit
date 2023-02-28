===========================
Visualize a quantum circuit
===========================

This guide shows how to get a visual representation of a quantum circuit.

Build the circuit
=================

The first step is to create the circuit.

.. testcode::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))

Visualize the circuit
=====================

There are three different ways to visualize a circuit. You can use

* The ``print()`` function.
* The :meth:`~.QuantumCircuit.draw` method.
* The :func:`~.circuit_drawer` function.

``print()``
-----------

If you call the ``print()`` function on a :class:`~.QuantumCircuit` object, you will get an `ASCII art version <https://en.wikipedia.org/wiki/ASCII_art>`_ of the circuit diagram.

.. testcode::

    print(qc)

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         ┌───┐     ┌─┐   
    q_0: ┤ H ├──■──┤M├───
         ├───┤┌─┴─┐└╥┘┌─┐
    q_1: ┤ H ├┤ X ├─╫─┤M├
         ├───┤└┬─┬┘ ║ └╥┘
    q_2: ┤ H ├─┤M├──╫──╫─
         └───┘ └╥┘  ║  ║ 
    c: 3/═══════╩═══╩══╩═
                2   0  1 

:meth:`~.QuantumCircuit.draw`
---------------------------------------------

You can also call the :meth:`~.QuantumCircuit.draw` method on a :class:`~.QuantumCircuit` object to visualize it. If you don't specify any arguments you will get a plain text representation of the circuit.

.. code-block:: python

    qc.draw()

.. code-block:: text

         ┌───┐     ┌─┐   
    q_0: ┤ H ├──■──┤M├───
         ├───┤┌─┴─┐└╥┘┌─┐
    q_1: ┤ H ├┤ X ├─╫─┤M├
         ├───┤└┬─┬┘ ║ └╥┘
    q_2: ┤ H ├─┤M├──╫──╫─
         └───┘ └╥┘  ║  ║ 
    c: 3/═══════╩═══╩══╩═
                2   0  1  

However, if you change the ``output`` argument, you can get other different renderings. The available options for this argument are:

* ``'text'``: renders the circuit with ASCII art. It's the default option.
* ``'mpl'``: uses `matplotlib <https://matplotlib.org/>`_ to render the circuit.
* ``'latex'``: uses :math:`\LaTeX` to render the circuit. It requires a full `LaTeX <https://latex.org/forum/>`_ distribution and the package ``pdflatex``.
* ``'latex_source'``: outputs the :math:`\LaTeX` source code that creates the ``'latex'`` rendering of the circuit.

Because this optional or keyword argument is actually the first of this method, one can type ``qc.draw(option)`` instead of ``qc.draw(output=option)``.

.. note::
    By default, the :meth:`~.QuantumCircuit.draw` method returns the rendered image as an object and does not output anything. The exact class returned depends on the output specified: ``'text'`` (the default) returns a ``TextDrawer`` object, ``'mpl'`` returns a `matplotlib.figure.Figure <https://matplotlib.org/stable/api/figure_api.html?highlight=figure#matplotlib.figure.Figure>`_ object, and ``'latex'`` returns a ``PIL.Image`` object. Having the return types enables modifying or directly interacting with the rendered output from the drawers. Jupyter notebooks understand these return types and render them for us in this guide, but when running outside of Jupyter, you do not have this feature automatically. However, the :meth:`~.QuantumCircuit.draw` method has optional arguments to display or save the output. When specified, the ``filename`` kwarg takes a path to which it saves the rendered output. Alternatively, if you're using the ``'mpl'`` or ``'latex'`` outputs, you can leverage the ``interactive`` kwarg to open the image in a new window (this will not always work from within a notebook).


``'mpl'``
^^^^^^^^^

.. code-block:: python

    qc.draw('mpl')

.. plot::

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))
    qc.draw("mpl")


``'latex_source'``
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    qc.draw('latex_source')

.. code-block:: text

    '\\documentclass[border=2px]{standalone}\n\n\\usepackage[braket, qm]{qcircuit}\n\\usepackage{graphicx}\n\n\\begin{document}\n\\scalebox{1.0}{\n\\Qcircuit @C=1.0em @R=0.2em @!R { \\\\\n\t \t\\nghost{{q}_{0} :  } & \\lstick{{q}_{0} :  } & \\gate{\\mathrm{H}} & \\ctrl{1} & \\meter & \\qw & \\qw & \\qw\\\\\n\t \t\\nghost{{q}_{1} :  } & \\lstick{{q}_{1} :  } & \\gate{\\mathrm{H}} & \\targ & \\qw & \\meter & \\qw & \\qw\\\\\n\t \t\\nghost{{q}_{2} :  } & \\lstick{{q}_{2} :  } & \\gate{\\mathrm{H}} & \\meter & \\qw & \\qw & \\qw & \\qw\\\\\n\t \t\\nghost{\\mathrm{{c} :  }} & \\lstick{\\mathrm{{c} :  }} & \\lstick{/_{_{3}}} \\cw & \\dstick{_{_{\\hspace{0.0em}2}}} \\cw \\ar @{<=} [-1,0] & \\dstick{_{_{\\hspace{0.0em}0}}} \\cw \\ar @{<=} [-3,0] & \\dstick{_{_{\\hspace{0.0em}1}}} \\cw \\ar @{<=} [-2,0] & \\cw & \\cw\\\\\n\\\\ }}\n\\end{document}'


:func:`~.circuit_drawer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to use a self-contained function instead of a :class:`~.QuantumCircuit` method to draw your circuit, you can do it with :func:`~.circuit_drawer` from :mod:`qiskit.visualization`. It has the exact same behavior as the :meth:`~.QuantumCircuit.draw` method above, except that it requires the circuit to be included as an argument.

.. note::
    In Qiskit Terra :math:`\leq 0.7`, the default behavior for the :func:`~.circuit_drawer` function is to use the ``'latex'`` output backend, and in :math:`0.6.x` that includes a fallback to ``'mpl'`` if ``'latex'`` fails for any reason. Starting with release :math:`> 0.7`, the default changes to the ``'text'`` output.


.. code-block:: python

    from qiskit.visualization import circuit_drawer

    circuit_drawer(qc, output='mpl')

.. plot::

    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer

    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    qc.cx(0, 1)
    qc.measure(range(3), range(3))
    circuit_drawer(qc, output='mpl')