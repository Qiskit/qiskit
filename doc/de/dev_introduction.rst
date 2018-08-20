Aufbau
======

Programmierschnittstelle
------------------------

Der *qiskit* Ordner stellt das Python Hauptmodul dar und beinhaltet die
Objekte für die Programmierschnittstelle
:py:class:`QuantumProgram <qiskit.QuantumProgram>`,
:py:class:`QuantumRegister <qiskit.QuantumRegister>`,
:py:class:`ClassicalRegister <qiskit.ClassicalRegister>`,
und :py:class:`QuantumCircuit <qiskit.QuantumCircuit>`.

Der Anwender erzeugt auf höchster Ebene ein *QuantumProgram* zum Erstellen,
Übersetzen und Ausführen von Quantum Circuits. Jeder *QuantumCircuit*
umfasst einen Satz an Datenregistern, vom Typ *QuantumRegister* oder
*ClassicalRegister*. Methoden dieser Objekte werden verwendet, um durch
Instruktionen die Circuits zu definieren. Ein *QuantumCircuit* kann dann
**OpenQASM** Code generieren, der durch andere Komponenten im *qiskit*
Ordner verarbeitet werden kann.

Der Ordner :py:mod:`Erweiterungen <qiskit.extensions>` erweitert
Quantum Circuits bei Bedarf, um andere Menge von Gattern und Algorithmen zu
unterstützen. Zur Zeit definiert die Erweiterung
:py:mod:`standard <qiskit.extensions.standard>` typische Quantengatter und es
existieren die beiden zusätzlichen Erweiterungen
:py:mod:`qasm_simulator_cpp <qiskit.extensions.simulator>` und
:py:mod:`quantum_initializer <qiskit.extensions.quantum_initializer>`.

Interne Module
--------------

Der *qiskit* Ordner beinhaltet auch interne Module, die noch in Entwicklung
sind:

- ein :py:mod:`qasm <qiskit.qasm>` Modul zum Parsen von **OpenQASM**
  Circuits
- ein :py:mod:`unroll <qiskit.unroll>` Modul zum Interpretieren und “Ausrollen”
  **OpenQASM** in eine Menge von Basisgattern (zum Erweitern von Gattern,
  Unterroutinen und Schleifen)
- ein :py:mod:`dagcircuit <qiskit.dagcircuit>` Modul zum Arbeiten mit
  Cirucits als Graphen
- ein :py:mod:`mapper <qiskit.mapper>` Modul zum Mappen von Circuits
  mit jeder-zu-jeder Verbindungen, um auf Geräten mit fixen Verbindungen
  lauffähigen Code zu generieren
- ein :py:mod:`backends <qiskit.backends>` Modul, das die Quantum Circuit
  Simulatoren enthält
- einen Ordner *tools*, der Methoden für Applikationen, Analyse und
  Visualisierung beinhaltet

Quantum Circuit werden wie folgt durch die Komponenten gereicht. Die
Programmierschnittstelle wird verwendet, um **OpenQASM** Circuits zu
generieren, als Texte oder als *QuantumCircuit* Objekte. **OpenQASM**
Quellcode, als eine Datei oder String wird in ein *Qasm* Objekt
übergeben, dessen Parser Methode einen abstrakten Syntaxbaum, (abstract syntax
tree, **AST**) erzeugt. Der **AST** wird von einem *Unroller* verarbeitet,
der an ein *UnrollerBackend* angeschlossen ist.
Es gibt ein *PrinterBackend* zur Ausgabe von Text, ein *JsonBackend* zum
Erzeugen von Input für die Simulatoren und Experiment Backends, ein
*DAGBackend* zur Konstruktion von *DAGCircuit* Objekten und
ein *CircuitBackend* zur Produktion von *QuantumCircuit* Objekten. Das
*DAGCircuit* Objekt stellt einen “ausgerollten” **OpenQASM** Circuit als
einen gerichteten azyklischen Graphen (directed acyclic graph, DAG) dar. Der
*DAGCircuit* stellt Methoden zur Verfügung zur Repräsentierung,
Transformierung und Berechnung von Eigenschaften eines Circuit und gibt die
Ergebnisse wiederum als **OpenQASM** aus. Der gesamte Ablauf wird vom
*mapper* Modul verwendet, um einen Circuit umzuschreiben und ihn auf einem
Gerät mit festen Kopplungen gemäß dem *CouplingGraph* ausführbar zu machen. Die
Struktur dieser Komponenten kann sich unter Umständen ändern.

Die Darstellung als Circuit und wie diese zur Zeit ineinander transformiert
werden, sind in folgendem Diagramm zusammengefasst:


.. image:: ../../images/circuit_representations.png
    :width: 600px
    :align: center

Mehrere *Unroller* Backends und deren Ausgaben sind hier zusammengefasst:


.. image:: ../../images/unroller_backends.png
    :width: 600px
    :align: center


Protokollierung
---------------

Das SDK verwendet die `Standard Python "logging" Bibliothek
<https://docs.python.org/3/library/logging.html>`_ zur Ausgabe von diversen
Nachrichten mit Hilfe der Familie von "`qiskit.*`" Loggern, und hält sich
an die Konventionen von Logging Level:

.. tabularcolumns:: |l|L|

+--------------+----------------------------------------------+
| Level        | Wird wann verwendet                          |
+==============+==============================================+
| ``DEBUG``    | Detaillierte Informationen, typischerweise   |
|              | nur bei der Diagnose von Problemen von       |
|              | Interesse.                                   |
+--------------+----------------------------------------------+
| ``INFO``     | Bestätigung, dass alles wie erwartet         |
|              | funktioniert.                                |
+--------------+----------------------------------------------+
| ``WARNING``  | Hinweis, dass etwas Unerwartetes passiert    |
|              | ist oder als Anzeige von irgendeinem Problem |
|              | in naher Zukunft (z.B. 'disk space low').    |
|              | Die Software funktioniert weiterhin wie      |
|              | erwartet.                                    |
+--------------+----------------------------------------------+
| ``ERROR``    | Aufgrund eines schwerwiegenderen Problems    |
|              | konnte die Software eine bestimmte Funktion  |
|              | nicht durchführen.                           |
+--------------+----------------------------------------------+
| ``CRITICAL`` | Ein schwerwiegender Fehler ist aufgetreten,  |
|              | das Programm selbst kann unter Umständen     |
|              | nicht weiter laufen.                         |
+--------------+----------------------------------------------+


Zur bequemeren Verwendung bietet
:py:class:`QuantumProgram <qiskit.QuantumProgram>` zwei Methoden an
(:py:func:`enable_logs() <qiskit.QuantumProgram.enable_logs>` und
:py:func:`disable_logs() <qiskit.QuantumProgram.disable_logs>`), die den
Handler und den Level vom `qiskit` Logger modifizieren. Das Verwenden dieser
Methoden kann Konflikte mit den globalen Einstellungen des Logging Setups Ihrer
Python Umgebung erzeugen. Bitte beachten Sie dies, wenn Sie eine
Applikation auf dem Qiskit SDK aufbauend entwickeln.

Die Konvention zur Ausgabe einer Logging Nachricht schreibt vor, im Modul
eine globale Variable mit Namen **logger** zu deklarieren, die den Logger mit
dem Namen des Moduls **__name__** beinhaltet. Dieses Objekt soll dann zum
Absenden von Nachrichten verwendet werden. Zum Beispiel für das Modul
`qiskit/some/module.py`:

.. code-block:: python

   import logging

   logger = logging.getLogger(__name__)  # logger for "qiskit.some.module"
   ...
   logger.info("This is an info message)


Testen
------

Das SDK verwendet das `standard Python "unittest" Framework
<https://docs.python.org/3/library/unittest.html>`_ zum Testen von
verschiednen Komponenten und Funktionalitäten.

Da das Qiskit Build-System auf CMake basiert, muss ein so genannter
"out-of-source" Build vor dem Ausführen der Tests durchgeführt werden. Dies
bedeutet einfach, dass folgende Befehle ausgeführt werden müssen:

Linux und Mac:

.. code-block:: bash

    $ mkdir out
    $ cd out
    out$ cmake ..
    out$ make

Windows:

.. code-block:: bash

    C:\..\> mkdir out
    C:\..\> cd out
    C:\..\out> cmake -DUSER_LIB_PATH=C:\path\to\mingw64\lib\libpthreads.a -G "MinGW Makefiles" ..
    C:\..\out> make

Dies wird alle notwendigen Binärdateien für Ihre spezifische Platform
generieren.

Um die Tests auszuführen, ist ein ``make test`` Ziel definiert. Die
Ausführung der Tests (durch das make Ziel genauso wie beim manuellen Aufruf)
berücksichtigt die ``LOG_LEVEL`` Umgebungsvariable. Wenn vorhanden, wird
eine ``.log`` Datei im Test Ordner erzeugt mit der Ausgabe der Logging
Aufrufe, die auch auf stdout ausgegeben werden. Sie können die Verbosität
über den Inhalt der Variable einstellen, zum Beispiel:

Linux und Mac:

.. code-block:: bash

    $ cd out
    out$ LOG_LEVEL="DEBUG" ARGS="-V" make test

Windows:

.. code-block:: bash

    $ cd out
    C:\..\out> set LOG_LEVEL="DEBUG"
    C:\..\out> set ARGS="-V"
    C:\..\out> make test

Zum händischen Ausführen eines einfachen Python Tests muss der Ordner nicht
auf ``out`` gewechselt werden. Es reicht folgender Befehl:

Linux und Mac:

.. code-block:: bash

    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/test_apps.py
