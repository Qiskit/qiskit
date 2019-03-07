# Qiskit Terra (양자정보과학 키트 - 키스킷 - 테라)


[![PyPI](https://img.shields.io/pypi/v/qiskit.svg)](https://pypi.python.org/pypi/qiskit)
[![Build Status](https://travis-ci.org/Qiskit/qiskit-terra.svg?branch=master)](https://travis-ci.org/Qiskit/qiskit-terra)
[![Build Status IBM Q](https://travis-matrix-badges.herokuapp.com/repos/Qiskit/qiskit-terra/branches/master/8)](https://travis-ci.org/Qiskit/qiskit-terra)

양자정보과학 키트(The Quantum Information Science Kit), 줄여서 **Qiskit**(키스킷)은 양자정보용 응용 프로그램을 개발하기 위해 만들어진 소프트웨어 개발용 키트(SDK)입니다. **Qiskit**은 노이즈가 많은 중간 크기의 양자컴퓨터(NISQ -- Noisy-Intermediate Scale Quantum --computers)와 같이 쓰일 수 있습니다.

Qiskit은 양자전산을 가능하게 하는 여러가지 요소들로 이루어져 있습니다. 이 폴더는 Terra(땅) - Qiskit이 세워질 수 있게 하는 기반을 제공하는 요소입니다.
(지세한 내용은 이 곳의 [개요](https://medium.com/qiskit/qiskit-and-its-fundamental-elements-bcd7ead80492)를 참고하세요).

## Qiskit의 설치방법

Qiskit 설치에 PIP (파이썬 패키지 매니저)툴을 이용하실 것을 권장합니다:

```bash
pip install qiskit
```

PIP으로 Qiskit을 설치하시면 모든 필수 프로그램들이 같이 자동으로 설치가 되기 때문에, 항상 검증된 최신 버전의 Qiskit을 쓰게 되실 수 있습니다. 

Qiskit을 쓰기 위해서는 적어도 [파이썬 Python 3.5 이상의 버전이](https://www.python.org/downloads/) 필요합니다. 그리고, 튜토리얼의 실행을 위해 [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html)의 사용을 추천합니다. 이러한 연유로 Qiskit을 쓰실 때는 [Anaconda 3](https://www.continuum.io/downloads) 파이썬 배포판의 사용을 권장합니다. 아나콘다에는 대부분의 필요한 툴이 미리 설치되어 있습니다. 

자세한 내용은 [Qiskit 설치방법](doc/install.rst)을 참고하세요. 소스코드로부터 Qiskit을 생성하는 법과 환경 설정 방법이 나와 있습니다.  

## 첫번째 양자 프로그램 만들기

Qiskit의 설치가 끝났습니다. 이제 Terra를 써볼 차례입니다. 우리는 이제 Qiskit Aer(에어)에서 시뮬레이션 한 양자회로를 실제 양자컴퓨터에 시험해 볼 준비가 다 되었습니다. 다음은 양자얽힘 상태를 만드는 간단한 예제입니다.  

```
$ python
```

```python
>>> from qiskit import *
>>> q = QuantumRegister(2)
>>> c = ClassicalRegister(2)
>>> qc = QuantumCircuit(q, c)
>>> qc.h(q[0])
>>> qc.cx(q[0], q[1])
>>> qc.measure(q, c)
>>> backend_sim = Aer.get_backend('qasm_simulator')
>>> result = execute(qc, backend_sim).result()
>>> print(result.get_counts(qc))
```

이 경우에 결과물은 다음과 같을 것입니다:

```python
{'counts': {'00': 513, '11': 511}}
```
이 예제는 [이곳](examples/python/hello_quantum.py)에서 예제 프로그램을 실제 IBMQ 양자컴퓨터에서 실행하는 방법과 함께 찾아보실 수 있습니다.

### 양자프로그램을 실제 양자컴퓨터에서 실행하기

Qiskit을 이용해서 만드신 양자프로그램을 **실제의 양자컴퓨터 소자**에서 구동해 보실 수 있습니다.  그러기 위해서는 Qiskit에 IBM Q 계정을 설정해야 합니다.  

#### IBMQ 계정과 보안증명

1. 먼저 _[IBM Q](https://quantumexperience.ng.bluemix.net) >계정_ 을 만듭니다.  

2. IBM Q 홈페이지에 로그 인 하신 후에 API 토큰을 받습니다. 토큰은 _My Account > Advanced > API Token_ 에서 찾아보실 수 있습니다. 이 API token은 IBM Q 의 양자컴퓨터들을 사용하실 때 필요합니다. 

3. Qiskit 프로그램을 IBM Q 백엔드에서 실행하실 때, 보안 증명을 위해 2번 단계에서 생성된 토큰을 `IBMQ.save_account()` 함수에 넘깁니다. 예를들어, 토큰을 `MY_API_TOKEN`이라는 변수에 저장했다면:

```python
from qiskit import IBMQ

IBMQ.save_account('MY_API_TOKEN')
``` 

  이 명령어로 IBM Q 로긴과 토큰이 로컬 저장매체에 저장되며 다음의 간단한 명령어로 토큰을 불러와 쓰실 수 있습니다:

```python
from qiskit import IBMQ

IBMQ.load_accounts()
```

  토큰을 저장매체에 저장하고 싶으지 않으시다면 다음의 명령어를 사용하시면 됩니다.  

```python
from qiskit import IBMQ

IBMQ.enable_account('MY_API_TOKEN')
``` 

  이 명령어를 사용하시면 토큰은 IBM Q 디바이스를 쓰시는 동안만 활성화 됩니다. 

**examples/python** 에 Terra를 IBM Q 실제 양자컴퓨터 소자에서 하는 예제들이 많이 나와 있습니다. [using_qiskit_terra_level_0.py](examples/python/using_qiskit_terra_level_0.py)부터 시작하셔서 단계적으로 올라가 보세요. 

## Qiskit에 기여하기 위한 가이드라인

Qiskit에 기여하고 싶으시다면 [contribution guidelines](.github/CONTRIBUTING.rst)을 보시기 바랍니다. 이 프로젝트는 Qiskit's [행동 강령 (code of conduct)](.github/CODE_OF_CONDUCT.rst)를 준수합니다. 프로젝트의 참여는 곧 모든 유저들이 행동강령을 준수한다는 것을 의미합니다. 

버그와 그 밖에 요청사항이 있으시면 [GitHub issues](https://github.com/Qiskit/qiskit-terra/issues)를 이용하세요. 토론은 Qiskit [slack](https://qiskit.slack.com)채널을 이용하시면 됩니다. Qiskit Slack채널에 가입하시려면 [링크](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk)를 클릭하세요.질문은 [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit)에서 하시면 됩니다. 

### Next Steps

이제 여러분은 [Qiskit 튜토리얼](https://github.com/Qiskit/qiskit-tutorial)에 있는 다른 예제들까지도 실행할 수 있는 준비와 환경 구성을 마쳤습니다. 

## Authors

Qiskit Terra는 [많은 사람들](https://github.com/Qiskit/qiskit-terra/graphs/contributors)의 노력과 기여로 계속 발전해 나가고 있습니다. 
