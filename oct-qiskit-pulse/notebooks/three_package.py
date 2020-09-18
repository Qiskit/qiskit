# %%
from src.QOCInstructionScheduleMap import *
from src.QutipOptimizer import *
from qiskit import IBMQ
from qiskit.circuit.quantumcircuit import QuantumCircuit
import qiskit
from qiskit.compiler.assemble import assemble
from qiskit.tools.monitor.job_monitor import job_monitor
from qiskit.circuit.library import XGate
from src.helper import qubit_distribution
# %%
# subsystem_list=[0,1]
subsystem_list = [0]
IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
backend = provider.get_backend('ibmq_valencia')
config = backend.configuration()

# %%
# backend.configuration().hamiltonian['vars']['omegad0'] = backend.configuration().hamiltonian['vars']['omegad0'] *2
grape_optimizer = QutipOptimizer(backend, n_ts=64, p_type='LIN', two_level=True)

builtin_instructions = backend.defaults().instruction_schedule_map
grape_inst_map = QOCInstructionScheduleMap.from_inst_map(grape_optimizer,
                                                         builtin_instructions,
                                                         ['measure'])

# %%
circ = QuantumCircuit(1, 1)
# circ.x(1)
circ.x(0)
# circ.cnot(0,1)
circ.measure(0, 0)
# circ.measure(1,1)

# %%
grape_schedule = qiskit.schedule(circ,
                                 inst_map=grape_inst_map,
                                 meas_map=backend.configuration().meas_map)
grape_schedule.draw(plot_range=[0, 200])
# %%
real_qobj = assemble(grape_schedule, backend=backend,
                     meas_level=2,
                     meas_return='single',
                     shots=1024)
real_job = backend.run(real_qobj)
# %%
job_monitor(real_job)
# %%
qubit_distribution(real_job.result().get_counts())
