QISKit入門
==========

:py:class:`QuantumProgram <qiskit.QuantumProgram>` オブジェクトがコードを書く際の起点になります。
QuantumProgramは量子回路と量子レジスターと古典レジスターで構成されます。
IBM Q Experienceを使ったことがある人にとって、量子回路はComposerの画面で設計するもののことです。
QuantumProgramのメソッドは量子回路を実機やシミュレーターのバックエンドに転送し、実行の結果を得て、
さらなる分析を行うことができます。

シミュレーター上で量子回路を設計して実行するには、以下のようにします。

.. code-block:: python

   from qiskit import QuantumProgram
   qp = QuantumProgram()

   qr = qp.create_quantum_register('qr', 2)
   cr = qp.create_classical_register('cr', 2)
   qc = qp.create_circuit('Bell', [qr], [cr])

   qc.h(qr[0])
   qc.cx(qr[0], qr[1])
   qc.measure(qr[0], cr[0])
   qc.measure(qr[1], cr[1])

   result = qp.execute('Bell')
   print(result.get_counts('Bell'))

:code:`get_counts` メソッドは {計算基底:出現回数} の辞書オブジェクトを出力します。
計算基底とは従来のビット列のことです。

.. code-block:: python

    {'00': 531, '11': 493}

量子プロセッサー
----------------

ユーザーはQASMで記述した量子回路をIBM Q Experience (QX)のクラウドプラットホームを通じて、
実機の量子コンピューター（量子プロセッサー）で実行することができます。
現在以下のチップが利用可能です:

-   `ibmqx2: 5-qubit backend <https://ibm.biz/qiskit-ibmqx2>`__
-   `ibmqx3: 16-qubit backend <https://ibm.biz/qiskit-ibmqx3>`__
-   `ibmqx4: 5-qubit backend <https://ibm.biz/qiskit-ibmqx4>`__
-   `ibmqx5: 16-qubit backend <https://ibm.biz/qiskit-ibmqx5>`__

最新の実機の詳細については
`IBM Q Experience バックエンド情報 <https://github.com/Qiskit/ibmqx-backend-information>`_ を参照してください。

.. include:: example_real_backend.rst

プロジェクト構成
----------------

Pythonのプログラム例は *examples* ディレクトリに、
テストスクリプトは *test* ディレクトリにあります。
*qiskit* ディレクトリがSDKのメインモジュールです。
