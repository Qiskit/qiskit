======================
Installation und Setup
======================

Installation
============

1. Abhängigkeiten
-----------------

Um Qiskit verwenden zu können benötigt man mindestens `Python 3.5 oder höher
<https://www.python.org/downloads/>`__. `Jupyter Notebooks <https://jupyter
.readthedocs.io/de/latest/install.html>`__ wird weiters empfohlen für die
Verwendung von den Beispielen in `tutorials`_.

Deshalb empfehlen wir die Installation der `Anaconda 3  <https://www
.continuum.io/downloads>`__ Python Distribution, die alle Abhängigkeiten
vorinstalliert mitliefert.


2. Installation
---------------

Die empfohlene Methode, um Qiskit zu installieren, ist durch die Verwendung des
PIP Tools (Python Paketmanager):

.. code:: sh

    pip install qiskit

Dieser Befehl wird den neuesten stabilen Release mitsamt allen Abhängigkeiten
installieren.

.. _qconfig-setup:

3. API Token und QE Anmeldedaten konfigurieren
----------------------------------------------

-  Erstellen Sie einen `IBM Q experience
   <https://quantumexperience.ng.bluemix.net>`__ Account falls Sie nicht bereits
   einen besitzen.
-  Holen Sie sich einen API Token von der IBM Q experience Webseite unter “My
   Account” > “Personal Access Token”
-  Der API Token muss in einer Datei gespeichert werden mit dem Namen
   ``Qconfig.py``. Als Beispiel haben wir eine Standardversion dieser Datei
   angelegt, die Sie als Referenz verwenden können: `Qconfig.py.default`_. Nach
   dem Herunterladen dieser Datei kopieren Sie diese in den Ordner von dem Sie
   das Qiskit SDK aufrufen werden (unter Windows ersetzen Sie ``cp`` mit
   ``copy``):

.. code:: sh

    cp Qconfig.py.default Qconfig.py

-  Öffnen Sie die Datei ``Qconfig.py``, entfernen Sie das ``#`` Symbol am
   Zeilenanfang des API Tokens und kopieren und fügen Sie den API Token zwischen
   den Anführungszeichen in dieser Zeile ein. Speichern und schließen Sie
   diese Datei.

Eine korrekte und voll konfigurierte ``Qconfig.py`` Datei würde
beispielsweise so aussehen:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api'
    }

-  Wenn Sie Zugriff auf die IBM Q Features haben, müssen Sie auch die Werte
   für ``hub``, ``group`` und ``project`` konfigurieren. Dies können Sie durch
   Anfügen der Werte von der Webseite Ihres IBM Q Kontos an die ``config``
   Variable bewerkstelligen.

Eine korrekte und voll konfigurierte ``Qconfig.py`` Datei würde für IBM Q
Benutzer so aussehen:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',
        # The following should only be needed for IBM Q users.
        'hub': 'MY_HUB',
        'group': 'MY_GROUP',
        'project': 'MY_PROJECT'
    }

Jupyter basierte Tutorials installieren
=======================================

Das Qiskit Projekt stellt eine Sammlung an Tutorials in Form von Jupyter
Notebooks zur Verfügung. Dabei handelt es sich um Webseiten, die Zellen von
eingebundenem Python Code beinhalten. Nähere Informationen finden Sie dazu in
`tutorials repository`_.


Problembehebung
===============

Die Schritte zur Installation in diesem Dokument setzen ein Vorwissen über
die Python Umgebung und Ihrem individuellem Setup voraus (zum Beispiel eine
Standard Python Installation, ``virtualenv`` oder Anaconda). Bitte verwenden
Sie die jeweilige Dokumentation für Anleitungen ihrem Setup betreffend.

Abhängig von Ihrem System und Setup kann ein angefügtes ``sudo -H`` vor ``pip
install`` notewndig sein.

Um auf die neueste Qiskit Version upzudaten verwenden Sie bitte folgenden
Befehl:

.. code:: sh

    pip install -U --no-cache-dir qiskit

Für zusätzliche Tipps zur Problembehandlung, verwenden Sie bitte die `Qiskit
troubleshooting page <https://github.com/Qiskit/qiskit-terra/wiki/Qiskit-Troubleshooting>`_ auf dem GitHub Wiki
des Projektes.

.. _tutorials: https://github.com/Qiskit/qiskit-tutorial
.. _tutorials repository: https://github.com/Qiskit/qiskit-tutorial
.. _documentation for contributors: https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst
.. _Qconfig.py.default: https://github.com/Qiskit/qiskit-terra/blob/stable/Qconfig.py.default