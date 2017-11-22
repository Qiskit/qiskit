import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from qiskit.transpiler import StageBase, StageInputOutput, StageError
import qiskit.qasm as qasm
import qiskit.unroll as unroll

class TransformStage(StageBase):
    def __init__(self):
        pass

    def get_name(self, name):
        return 'TransformStage'

    def handle_request(self, input):
        if not self._check_preconditions(input):
            return input

        output_format = input.get('format')
        dag_circuit = input.get('dag_circuit')

        if output_format == 'dag':
            transformed_circuit = dag_circuit
        elif output_format == 'json':
            input.insert('unroll_backend_target', 'json')
        elif output_format == 'qasm':
            transformed_circuit = dag_circuit.qasm()
        else:
            raise StageError('Unrecognized circuit format: {}'.format(
                    output_format))

        input.result = transformed_circuit

        return input

    def _check_preconditions(self, input):
        if not isinstance(input, StageInputOutput):
            raise StageError('Input instance not supported!')

        if not input.exists(['dag_circuit','format'])
            return False

        return True


    def _dag2json(input, dag_circuit, basis_gates='u1,u2,u3,cx,id'):
        """Make a Json representation of the circuit.

        Takes a circuit dag and returns json circuit obj. This is an internal
        function.

        Args:
            dag_ciruit (dag object): a dag representation of the circuit.
            basis_gates (str): a comma seperated string and are the base gates,
                                which by default are: u1,u2,u3,cx,id

        Returns:
            the json version of the dag
        """
        # TODO: Jay: I think this needs to become a method like .qasm() for the DAG.
        try:
            circuit_string = dag_circuit.qasm(qeflag=True)
        except TypeError:
            circuit_string = dag_circuit.qasm()

        # TODO Move this code to another Stage ... or reuse UnrollStage with
        # different logic.
        
        unroller = unroll.Unroller(qasm.Qasm(data=circuit_string).parse(),
                                   unroll.JsonBackend(basis_gates.split(",")))
        json_circuit = unroller.execute()
        return json_circuit
