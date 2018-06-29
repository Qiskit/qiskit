QISKit Überblick
================

Philosophie
-----------

QISKit umfasst eine Sammlung von Software, mit deren Hilfe man Quantum
Circuits geringer Tiefe bearbeiten und in naher Zukunft Applikationen
und Experimente für einen Quantencomputer bauen kann. In QISKit besteht
ein Quantum Program aus einem Array von Quantum Circuits. Der Programm
Workflow setzt sich zusammen aus drei Stufen: Build, Compile und Run.
Build erlaubt verschiedene Quantum Circuits aufzubauen, die das zu lösende
Problem repräsentieren. Compile schreibt diese um, sodass sie auf
unterschiedlichen Backends lauffähig sind (z.B. auf Simulatoren bzw.
auf echten Quantenchips mit unterschiedlichem Volumen, Größe, Fidelity, usw.).
Run startet die eigentliche Berechnung als Job. Nachdem der Job
durchgelaufen ist, werden die auszulesenden Daten gesammelt. Die Methoden,
die zum Zusmmensetzen der Ausgabedaten verwendet werden, hängen vom
jeweiligen Programm ab. Dies ergibt entweder die erwartete Antwort oder
ermöglicht es, das Programm für die nächste Instanz umzuschreiben und zu
verbessern.

Projekt Übersicht
-----------------
Das QISKit Projekt besteht aus:

* `QISKit core <https://github.com/QISKit/qiskit-core>`_: Das Python
  Software Development Kit dient zum Schreiben von Quantencomputer
  Experimenten, Programmen und Anwendungen.

* `QISKit API <https://github.com/QISKit/qiskit-api-py>`_: Ein kleiner
  Python Wrapper um die Quantum Experience HTTP API, der es ermöglicht
  sich mit dem Quantum Experience Server zu verbinden und Quantenpgoramme
  auszuführen.

* `QISKit OpenQASM <https://github.com/QISKit/qiskit-openqasm>`_:
  Beinhaltet Spezifikationen, Beispiele, Dokumentation und Tools
  für die OpenQASM Repräsentation.

* `QISKit Tutorial <https://github.com/QISKit/qiskit-tutorial>`_: Eine
  Sammlung von beispielhaften Jupyter Notebooks, die Funktionen aus QISKit
  verwenden.
