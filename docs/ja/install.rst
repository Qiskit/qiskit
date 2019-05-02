==========================
インストールとセットアップ
==========================

インストール
============

1. ツールの入手
---------------

QISKitを利用するには少なくとも `Python 3.5か以降 <https://www.python.org/downloads/>`__ と
`Jupyter Notebooks <https://jupyter.readthedocs.io/en/latest/install.html>`__ を
インストールしておく必要があります。
(後者はチュートリアルで対話的に操作することをお勧めします)。

一般ユーザーにQISKitが依存する多くのライブラリが含まれている
`Anaconda 3 <https://www.continuum.io/downloads>`__ という
Python ディストリビューションをお勧めします。

Mac OS Xのユーザーの場合はXcodeが役に立ちます: https://developer.apple.com/xcode/

QISKitに貢献したいか拡張したいユーザーにはGitもインストールする必要があります: https://git-scm.com/download/.


2. PIP インストール
-------------------

QISKitをインストールする最も簡単な方法はPIP tool(Pythonのパッケージマネージャー)を利用することです。

.. code:: sh

    pip install qiskit

3. レポジトリのインストール
---------------------------

それ以外にもローカルにQiskit SDKのレポジトリのクローンを作成してそのディレクトリで作業するのも一般的です:

-  もしGitをインストールしている場合は、以下のコマンドを実行してください:

.. code:: sh

    git clone https://github.com/Qiskit/qiskit-terra
    cd qiskit-terra

- もしGitをインストールしていない場合は、
  `Qiskit SDK GitHub repo <https://github.com/Qiskit/qiskit-terra>`__ の
  "Clone or download"ボタンをクリックして、
  その後ダウンロードしたファイルを展開し、そのディレクトリーに移動し作業を開始します。

3.1 環境の設定
^^^^^^^^^^^^^^

QISKitを単独で機能するライブラリとして使うには全ての依存するライブラリもインストールする必要があります。

.. code:: sh

    # システムに依存しますが, 必要に応じて "sudo -H" をコマンドの前に追加します。
    pip install -r requires.txt

チュートリアルを利用するにはAnacondaの環境をセットアップして依存するライブラリをインストールします。:

-  LinuxかMac OS X (Xcodeインストール済み)の場合、以下のコマンドを実行します。

.. code:: sh

    make env

-  Mac OS X (Xcodeなし)の場合, 以下のコマンドを実行します:

.. code:: sh

    conda create -y -n QISKitenv python=3 pip scipy
    activate QISKitenv
    pip install -r requires.txt

-  Windowsの場合、Anacondaのプロンプトで以下のコマンドを実行します:

.. code:: sh

    .\make env


4. APIトークンの設定
--------------------

-  `IBM Q Experience <https://quantumexperience.ng.bluemix.net>`__
   のアカウントがない場合は作成します。
-  IBM Q Experienceのウェブサイトの“My Account” > “Personal Access Token”
   からAPIトークンを取得します。
-  Qconfig.pyというファイルにAPIトークンを書き込きます。
   まずデフォルトのQconfig.pyをコピーします。
   (Windowsの場合 ``cp`` を ``copy`` で置き換えます):

.. code:: sh

    cp Qconfig.py.default Qconfig.py

-  Qconfig.pyをエディターで開き、 ``#APItoken`` で始まる行の ``#`` を取り除き、
   あなたのAPIトークンを記入して保存します。

Jupyterのチュートリアルのインストール
=====================================

QISKitプロジェクトはチュートリアルをJupyterノートブックの形式で提供します。
ノートブックはPythonのコードが埋め込まれたウェブページのようなものです。
埋め込まれたコードを実行するには``Shift+Enter``を押すか、
ページ上部のツールバーを使います。
出力は即座にページの下に表示されます。多くの場合埋め込まれたコードは上から順に実行します。
チュートリアルを使いはじめるには以下の通りにします。


1.1 インストール
----------------
- チュートリアルのダウンロード: https://github.com/Qiskit/qiskit-tutorial/archive/master.zip
- zipファイルの展開
- ターミナルで"qiskit-tutorial-master"のディレクトリーに移動し、以下を実行する:

.. code:: sh

    jupyter notebook index.ipynb

チュートリアルに関する詳しい説明は
`qiskit-tutorial repository <https://github.com/Qiskit/qiskit-tutorial>`__
を参照してください。

FAQ
===

もし依存ライブラリを更新してエラーが発生した場合以下のコマンドを試してみてください:

- システムに依存しますが, 必要に応じて "sudo -H" をコマンドの前に追加してください。

.. code:: sh

    pip install -U --no-cache-dir IBMQuantumExperience

- 修正: 以下のコマンドを実行します。

.. code:: sh

    curl https://bootstrap.pypa.io/ez_setup.py -o - | python

プロジェクトのGitHubのWikiのQiskit troubleshootingのページにさらに情報があります。
