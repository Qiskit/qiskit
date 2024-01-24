from io import BytesIO

import dill

from qiskit.circuit import Bit


class QiskitPickler(dill.Pickler):
    def __init__(self, file=None, qubit_cache=None):
        super().__init__(file)
        if qubit_cache is None:
            self.qubit_cache = {}
        else:
            self.qubit_cache = qubit_cache

    def get_qubit_cache(self):
        return self.qubit_cache

    def persistent_id(self, obj):
        if isinstance(obj, Bit):
            self.qubit_cache[id(obj)] = obj
            return ("QBit", id(obj))
        else:
            return None


class QiskitUnpickler(dill.Unpickler):
    def __init__(self, file=None, qubit_cache=None):
        super().__init__(file)
        if qubit_cache is None:
            self.qubit_cache = {}
        else:
            self.qubit_cache = qubit_cache

    def get_qubit_cache(self):
        return self.qubit_cache

    def persistent_load(self, pid):
        type_tag, qubit_id = pid
        if type_tag == "QBit":
            return self.qubit_cache[qubit_id]
        else:
            raise dill.UnpicklingError("unsupported persistent object")


def dumps(obj, qubit_cache):
    file = BytesIO()
    QiskitPickler(file, qubit_cache).dump(obj)
    return file.getvalue()


def loads(obj_bin, qubit_cache):
    file = BytesIO(obj_bin)
    return QiskitUnpickler(file, qubit_cache).load()
