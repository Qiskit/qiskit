import copy

from qiskit._qiskiterror import QISKitError
from qiskit import RegisterSizeError

class Result(object):
    """ Result Class.

    Class internal properties.

    Methods to process the quantum program after it has been run

    Internal::

        qobj =  { -- the quantum object that was complied --}
        result = {
            "status": --status (string),
            "result":
                [
                    {
                    "data":
                        {  #### DATA CAN BE A DIFFERENT DICTIONARY FOR EACH BACKEND ####
                        "counts": {’00000’: XXXX, ’00001’: XXXXX},
                        "time"  : xx.xxxxxxxx
                        },
                    "status": --status (string)--
                    },
                    ...
                ]
            }
    """

    def __init__(self, qobj_result, qobj):
        self.__qobj = qobj
        self.__result = qobj_result

    def __str__(self):
        """Get the status of the run.

        Returns:
            the status of the results.
        """
        return self.__result['status']

    def __iadd__(self, other):
        """Append a Result object to current Result object.

        Arg:
            other (Result): a Result object to append.
        Returns:
            The current object with appended results.
        """
        if self.__qobj['config'] == other.__qobj['config']:
            if isinstance(self.__qobj['id'], str):
                self.__qobj['id'] = [self.__qobj['id']]
            self.__qobj['id'].append(other.__qobj['id'])
            self.__qobj['circuits'] += other.__qobj['circuits']
            self.__result['result'] += other.__result['result']
            return self
        else:
            raise QISKitError('Result objects have different configs and cannot be combined.')

    def __add__(self, other):
        """Combine Result objects.

        Note that the qobj id of the returned result will be the same as the
        first result.

        Arg:
            other (Result): a Result object to combine.
        Returns:
            A new Result object consisting of combined objects.
        """
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def _is_error(self):
         return self.__result['status'] == 'ERROR'

    def get_status(self):
        """Return whole qobj result status."""
        return self.__result['status']

    def circuit_statuses(self):
        """Return statuses of all circuits

        Return:
            List of status result strings.
        """
        return [circuit_result['status']
                for circuit_result in self.__result['result']]

    def get_ran_qasm(self, name):
        """Get the ran qasm for the named circuit and backend.

        Args:
            name (str): the name of the quantum circuit.

        Returns:
            A text version of the qasm file that has been run.
        """
        try:
            qobj = self.__qobj
            for index in range(len(qobj["circuits"])):
                if qobj["circuits"][index]['name'] == name:
                    return qobj["circuits"][index]["compiled_circuit_qasm"]
        except KeyError:
            raise QISKitError('No  qasm for circuit "{0}"'.format(name))

    def get_data(self, name):
        """Get the data of cicuit name.

        The data format will depend on the backend. For a real device it
        will be for the form::

            "counts": {’00000’: XXXX, ’00001’: XXXX},
            "time"  : xx.xxxxxxxx

        for the qasm simulators of 1 shot::

            'quantum_state': array([ XXX,  ..., XXX]),
            'classical_state': 0

        for the qasm simulators of n shots::

            'counts': {'0000': XXXX, '1001': XXXX}

        for the unitary simulators::

            'unitary': np.array([[ XX + XXj
                                   ...
                                   XX + XX]
                                 ...
                                 [ XX + XXj
                                   ...
                                   XX + XXj]]

        Args:
            name (str): the name of the quantum circuit.

        Returns:
            A dictionary of data for the different backends.

        Raises:
            If there's an error the function will throw a QISKitError or a
            RegisterSizeError.
        """
        if self._is_error():
            exception = self.__result['result']
            if isinstance(exception, RegisterSizeError):
                raise exception # Re-throw RegisterSizeError
            raise QISKitError(str(exception))

        try:
            qobj = self.__qobj
            for index in range(len(qobj['circuits'])):
                if qobj['circuits'][index]['name'] == name:
                    return self.__result['result'][index]['data']
        except (KeyError, TypeError):
            raise QISKitError('No data for circuit "{0}"'.format(name))

    def get_counts(self, name):
        """Get the histogram data of cicuit name.

        The data from the a qasm circuit is dictionary of the format
        {’00000’: XXXX, ’00001’: XXXXX}.

        Args:
            name (str): the name of the quantum circuit.
            backend (str): the name of the backend the data was run on.

        Returns:
            A dictionary of counts {’00000’: XXXX, ’00001’: XXXXX}.
        """
        try:
            return self.get_data(name)['counts']
        except KeyError:
            raise QISKitError('No counts for circuit "{0}"'.format(name))

    def get_names(self):
        """Get the circuit names of the results.

        Returns:
            A list of circuit names.
        """
        return [c['name'] for c in self.__qobj['circuits']]

    def average_data(self, name, observable):
        """Compute the mean value of an diagonal observable.

        Takes in an observable in dictionary format and then
        calculates the sum_i value(i) P(i) where value(i) is the value of
        the observable for state i.

        Args:
            name (str): the name of the quantum circuit
            obsevable (dict): The observable to be averaged over. As an example
            ZZ on qubits equals {"00": 1, "11": 1, "01": -1, "10": -1}

        Returns:
            a double for the average of the observable
        """
        counts = self.get_counts(name)
        temp = 0
        tot = sum(counts.values())
        for key in counts:
            if key in observable:
                temp += counts[key] * observable[key] / tot
        return temp
    