Provider interface to Sessions
==

```python
from qiskit_ibm_runtime import Session

with Session(service="estimator", params={'ansatzen': [A1, A2], 'observables': [O1, O2],...}) as sess1:
    sess1.write(request)
    rep = sess1.recv()
```

I'm imagining `Session` to be built on something like the nanomsg [req-rep Scalability Protocol](https://github.com/nanomsg/nanomsg/blob/master/rfc/sp-request-reply-01.txt), (Req on the client side, Rep on the server) using a binding similar to the [pynng](https://pynng.readthedocs.io/en/latest/) wrapper to nng. In that pattern the `Session` can have a "dialer" to reconnect if it gets sent to a cloud instance, and there can be multiple transports (TCP, UDP, websocket, etc), allthough the pynng implementation doesn't support pickling.

You can have exactly one request in flight and can send exactly one response. Requests should be idempotent. In order to do pipelining. You can make "contexts" from the sessions, so you can have req/rep per context:

```python
# thread 0, eg on lithops
with sess1.context("main loop") as c:
    # title is just for logging/documentation
    c.write(request)
    rep = c.recv()

#thread 1, eg on laptop
with sess1.context("monitoring loop") as c2:
    # title is just for logging/documentation
    c2.write(request)
    rep = c.recv()
```

Between the multiple contexts the Rep socket is supposed to round robin which requests it services.

Qiskit Interface to Estimator
==

```python

from ibm_provider import PauliEstimatorFactory

estimator_factory = PaulilEstimatorFactory(...)

def do_chemistry(..., estimator_factory):
    ...
    with estimator_factory.create(circuits=[psi1, psi2], observables=[H1, H2, H3], grouping=[[0,1], [2]], **backend_kwargs) as estimator:
        result = estimator.estimate(values, 0) # sub values -> params, evaluate
```

```python
class ExpectationValueFactory(ABC):

    @abstractmethod
    def create(self,
               observables: List[BaseOperator],
               circuits: List[QuantumCircuit],
               grouping: Optional[List[Tuple[int, int]]],
               exp_val: ExpectationValue = None) -> ExpectationValue:
        """
        Returns a new expectation value object based on the given observable, state, and mapping (if given).
        If not all of observable/state/mapping are given, but another expectation value, than the new expectation
        value is based on the given one and only overwrittes the new input.

        Args:
            observable: one or more observables to be evaluated in one or more states
            state: one or more states to evaluate the given observables
            mapping: how to combine given states and observables, if none is given, all combinations are evaluated
            exp_val: template for the new expectation value. Only the given arguments are changed.

        Returns:
            A new expectation value object.
        """
        pass


class PauliExpectationValueFactory(ExpectationValueFactory):

    def __init__(self, backend, shots=1024):
        super().__init__()
        self._backend = backend
        self._shots = shots

    def create(self, observable=None, state=None, mapping=None, exp_val=None):
        if exp_val:
            if observable is None:
                observable = exp_val.observable
            if state is None:
                state = exp_val.state
            if mapping is None:
                mapping = exp_val.mapping
        return PauliExpectationValue(observable, state, mapping, self._backend, self._shots)
```
