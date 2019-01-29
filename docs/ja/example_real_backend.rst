実機バックエンドの例
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from qiskit import QuantumProgram

    # 初めてのQuantumProgramオブジェクトのインスタンス生成
    Q_program = QuantumProgram()

    # APIトークンの設定
    # トークンは次のURLから入手可能です。 https://quantumexperience.ng.bluemix.net/qx/account,
    # "Personal Access Token"のセクションを参照してください。
    QX_TOKEN = "API_TOKEN"
    QX_URL = "https://quantumexperience.ng.bluemix.net/api"

    # APIをセットアップしてプログラムの実行。
    # APIトークンとQXのURLが必要です。
    Q_program.set_api(QX_TOKEN, QX_URL)

    # 2量子ビットの量子レジスターを生成して"qr"と名づけます。
    qr = Q_program.create_quantum_register("qr", 2)
    # 2ビットの古典レジスターを生成して"cr"と名づけます。
    cr = Q_program.create_classical_register("cr", 2)
    # 量子レジスター"qr"と古典レジスター"cr"を使った量子回路"qc"を生成ます。
    qc = Q_program.create_circuit("superposition", [qr], [cr])

    # Hゲートを量子ビット0に適用して量子重ね合わせを作ります。
    qc.h(qr[0])

    # "qr"の量子状態を観測し、その結果を"cr"に格納します。
    qc.measure(qr, cr)

    # 量子回路をコンパイルして実機のバックエンドibmqx4で実行します。
    result = Q_program.execute(["superposition"], backend='ibmqx4', shots=1024)

    # 結果を表示します。
    print(result)
    print(result.get_data("superposition"))
