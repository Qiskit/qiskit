###################################
Welcome to Qiskit's documentation!
###################################

Qiskit is an open-source framework for working with quantum computers
at the level of pulses, circuits, and algorithms.

A central goal of Qiskit is to build a software stack
that makes it easy for anyone to use quantum computers. However, Qiskit also aims
to facilitate research on the most important open issues facing quantum computation today.

You can use Qiskit to easily design experiments and run them on simulators and real
quantum computers.

Qiskit consists of four foundational elements:

- **Terra**: Composing quantum programs at the level of circuits and pulses with the
  code foundation
- **Aqua**: Building algorithms and applications
- **Ignis**: Addressing noise and errors
- **Aer**: Accelerating development via simulators, emulators, and debuggers

****************
Qiskit Elements
****************


:ref:`Qiskit Terra <Terra>`
   Terra, the ‘earth’ element, is the foundation on which the rest of Qiskit lies.
   Terra provides a bedrock for composing quantum programs at the level of circuits and pulses,
   to optimize them for the constraints of a particular device, and to manage the execution
   of batches of experiments on remote-access devices. Terra defines the interfaces
   for a desirable end-user experience, as well as the efficient handling of layers
   of optimization, pulse scheduling and backend communication.


:ref:`Qiskit Aer <Aer>`
   Aer, the ‘air’ element, permeates all Qiskit elements. To really speed up development
   of quantum computers we need better simulators, emulators and debuggers.  Aer helps
   us understand the limits of classical processors by demonstrating to what extent they
   can mimic quantum computation. Furthermore, we can use Aer to verify that current
   and near-future quantum computers function correctly. This can be done by stretching
   the limits of simulation, and by simulating the effects of realistic noise on
   the computation.


:ref:`Qiskit Ignis <Ignis>`
   Ignis, the ‘fire’ element, is dedicated to fighting noise and errors and to forging
   a new path. This includes better characterization of errors, improving gates, and computing
   in the presence of noise. Ignis is meant for those who want to design quantum error
   correction codes, or who wish to study ways to characterize errors through methods
   such as tomography, or even to find a better way for using gates by exploring
   dynamical decoupling and optimal control.


:ref:`Qiskit Aqua <Aqua>`
   Aqua, the ‘water’ element, is the element of life. To make quantum computing live up
   to its expectations, we need to find real-world applications. Aqua is where algorithms
   for quantum computers are built. These algorithms can be used to build applications
   for quantum computing. Aqua is accessible to domain experts in chemistry, optimization,
   finance and AI, who want to explore the benefits of using quantum computers as accelerators
   for specific computational tasks.


.. toctree::
  :maxdepth: 2
  :hidden:

  install
  getting_started
  terra/index
  aer/index
  ignis/index
  aqua/index
  development_strategy
  contributing_to_qiskit
  community
  release_notes
  faq
  API References <autodoc/qiskit>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
