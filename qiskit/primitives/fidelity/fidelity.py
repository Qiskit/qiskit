from abc import abstractmethod
import numpy as np

from qiskit import QiskitError, QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.circuit import ParameterVector

from typing import Optional, Callable, List, Union

from qiskit_machine_learning import QiskitMachineLearningError

SamplerFactory = Callable[[List[QuantumCircuit]], Sampler]


class BaseFidelity:
    """
    Implements the interface to calculate fidelities.
    """

    def __init__(
        self,
        left_circuit: Optional[QuantumCircuit] = None,
        right_circuit: Optional[QuantumCircuit] = None,
        sampler_factory: Optional[SamplerFactory] = None,
    ):
        """
        Initializes the class to evaluate the fidelities defined as the state overlap
            |<left_circuit(x), right_circuit(y)>|^2,
        where x and y are parametrizations of the circuits.
        Args:
            - left_circuit: (Parametrized) quantum circuit
            - right_circuit: (Parametrized) quantum circuit
            - sampler_factory: Optional partial sampler used as a backend
        Raises:
            - ValueError: left_circuit and right_circuit don't have the same number of qubits
        """
        if left_circuit is None or right_circuit is None:
            self._left_circuit = None
            self._right_circuit = None
        else:
            self.set_circuits(left_circuit, right_circuit)

        if sampler_factory is not None:
            self.sampler_from_factory(sampler_factory)
        else:
            self.sampler = None

    def set_circuits(self, left_circuit: QuantumCircuit, right_circuit: QuantumCircuit):
        """
        Fix the circuits for the fidelity to be computed of.
        Args:
            - left_circuit: (Parametrized) quantum circuit
            - right_circuit: (Parametrized) quantum circuit
        """
        if left_circuit.num_qubits != right_circuit.num_qubits:
            raise ValueError(
                f"The number of qubits for the left circuit ({left_circuit.num_qubits}) and right circuit ({right_circuit.num_qubits}) do not coincide."
            )
        # Assigning parameter arrays to the two circuits
        self._left_parameters = ParameterVector("x", left_circuit.num_parameters)
        self._left_circuit = left_circuit.assign_parameters(self._left_parameters)

        self._right_parameters = ParameterVector("y", right_circuit.num_parameters)
        self._right_circuit = right_circuit.assign_parameters(self._right_parameters)

    @abstractmethod
    def sampler_from_factory(self, sampler_factory: SamplerFactory):
        """
        Create a sampler instance from the sampler factory.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        values_left: Union[np.ndarray, List[np.ndarray]],
        values_right: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[float, List[float]]:
        """Compute the overlap of two quantum states bound by the parametrizations values_left and values_right.

        Args:
            values_left: Numerical parameters to be bound to the left circuit
            values_right: Numerical parameters to be bound to the right circuit

        Returns:
            The overlap of two quantum states defined by two parametrized circuits.
        """
        raise NotImplementedError


class Fidelity(BaseFidelity):
    """
    Calculates the fidelity of two quantum circuits by measuring the zero probability outcome.
    """

    def sampler_from_factory(self, sampler_factory: SamplerFactory):
        circuit = self._left_circuit.compose(self._right_circuit.inverse())
        circuit.measure_all()

        self.sampler = sampler_factory([circuit])

    def compute(
        self,
        values_left: Union[np.ndarray, List[np.ndarray]],
        values_right: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[float, List[float]]:
        if self._left_circuit is None or self._right_circuit is None:
            raise QiskitError(
                "The left and right circuits have to be set before computing the fidelity."
            )
        if self.sampler is None:
            raise QiskitError(
                "A sampler has to be instantiated by adding a sampler factory in order to compute the fidelity."
            )
        values_left = np.atleast_2d(values_left)
        values_right = np.atleast_2d(values_right)
        if values_left.shape[0] != values_right.shape[0]:
            raise ValueError(
                f"The number of left parameters (currently {values_left.shape[0]}) has to be equal to the number of right parameters (currently {values_right.shape[0]})"
            )
        values = np.hstack([values_left, values_right])
        result = self.sampler(
            circuit_indices=[0] * len(values), parameter_values=values
        )

        overlaps = [prob_dist.get(0, 0) for prob_dist in result.quasi_dists]
        return np.array(overlaps)
