import scipy
from qiskit.synthesis.unitary import qsd
from qiskit import transpile

nqubits = 3

dim = 2**nqubits
umat = scipy.stats.unitary_group.rvs(dim, random_state=1224)
circ = qsd.qs_decomposition(umat, opt_a1=True, opt_a2=True)

passes = []
def callback_func(**kwargs):
    t_pass = kwargs['pass_'].name()
    passes.append(t_pass)
ccirc = transpile(
    circ, basis_gates=["u", "cx", "qsd2q"], optimization_level=0, qubits_initially_zero=False, callback=callback_func
)
print("qsd passes", passes)

