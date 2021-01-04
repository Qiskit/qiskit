# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Variational Quantum Classifier algorithm."""

from typing import Optional, Callable, Dict, Union, Any
import warnings
import logging
import math
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, ParameterExpression

from qiskit.components.feature_maps import FeatureMap
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.utils.dataset_helper import (map_label_to_class_name,
                                         split_dataset_to_data_and_labels)
from ..variational_quantum_algorithm import VQAlgorithm
from ..optimizers import Optimizer
from ..variational_forms import VariationalForm
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class VQC(VQAlgorithm):
    """The Variational Quantum Classifier algorithm.

    Similar to :class:`QSVM`, the VQC algorithm also applies to classification problems.
    VQC uses the variational method to solve such problems in a quantum processor.  Specifically,
    it optimizes a parameterized quantum circuit to provide a solution that cleanly separates
    the data.

    .. note::

        The VQC stores the parameters of `var_form` and `feature_map` sorted by name to map the
        values provided by the optimizer to the circuit. This is done to ensure reproducible
        results, for example such that running the optimization twice with same random seeds yields
        the same result.

    """

    def __init__(
            self,
            optimizer: Optimizer,
            feature_map: Union[QuantumCircuit, FeatureMap],
            var_form: Union[QuantumCircuit, VariationalForm],
            training_dataset: Dict[str, np.ndarray],
            test_dataset: Optional[Dict[str, np.ndarray]] = None,
            datapoints: Optional[np.ndarray] = None,
            max_evals_grouped: int = 1,
            minibatch_size: int = -1,
            callback: Optional[Callable[[int, np.ndarray, float, int], None]] = None,
            quantum_instance: Optional[
                Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """
        Args:
            optimizer: The classical optimizer to use.
            feature_map: The FeatureMap instance to use.
            var_form: The variational form instance.
            training_dataset: The training dataset, in the format
                {'A': np.ndarray, 'B': np.ndarray, ...}.
            test_dataset: The test dataset, in same format as `training_dataset`.
            datapoints: NxD array, N is the number of data and D is data dimension.
            max_evals_grouped: The maximum number of evaluations to perform simultaneously.
            minibatch_size: The size of a mini-batch.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation.
                These are: the evaluation count, parameters of the variational form,
                the evaluated value, the index of data batch.
            quantum_instance: Quantum Instance or Backend

        Note:
            We use `label` to denotes numeric results and `class` the class names (str).

        Raises:
            AlgorithmError: Missing feature map or missing training dataset.
        """
        # VariationalForm is not deprecated on level of the VQAlgorithm yet as UCCSD still
        # derives from there, therefore we're adding a warning here
        if isinstance(var_form, VariationalForm):
            warnings.warn("""
            The {} object as input for the VQC is deprecated as of 0.7.0 and will
            be removed no earlier than 3 months after the release.
            You should pass a QuantumCircuit object instead.
            See also qiskit.circuit.library.n_local for a collection
            of suitable circuits.""".format(type(feature_map)),
                          DeprecationWarning, stacklevel=2)
        super().__init__(
            var_form=var_form,
            optimizer=optimizer,
            cost_fn=self.loss,
            quantum_instance=quantum_instance
        )
        self._batches = None
        self._label_batches = None
        self._batch_index = None
        self._eval_time = None
        self.batch_num = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)

        self._callback = callback

        if feature_map is None:
            raise AlgorithmError('Missing feature map.')
        if training_dataset is None:
            raise AlgorithmError('Missing training dataset.')
        self._training_dataset, self._class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        self._label_to_class = {label: class_name for class_name, label
                                in self._class_to_label.items()}
        self._num_classes = len(list(self._class_to_label.keys()))

        if test_dataset is not None:
            self._test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                  self._class_to_label)
        else:
            self._test_dataset = test_dataset

        if datapoints is not None and not isinstance(datapoints, np.ndarray):
            datapoints = np.asarray(datapoints)
            if len(datapoints) == 0:  # pylint: disable=len-as-condition
                datapoints = None
        self._datapoints = datapoints
        self._minibatch_size = minibatch_size

        self._eval_count = 0
        self._ret = {}  # type: Dict[str, Any]
        self._parameterized_circuits = None

        self.feature_map = feature_map

    def construct_circuit(self, x, theta, measurement=False):
        """Construct circuit based on data and parameters in variational form.

        Args:
            x (numpy.ndarray): 1-D array with D dimension
            theta (list[numpy.ndarray]): list of 1-D array, parameters sets for variational form
            measurement (bool): flag to add measurement

        Returns:
            QuantumCircuit: the circuit

        Raises:
            AlgorithmError: If ``x`` and ``theta`` share parameters with the same name.
        """
        # check x and theta do not have parameters of the same name
        x_names = [param.name for param in x if isinstance(param, ParameterExpression)]
        theta_names = [param.name for param in theta if isinstance(param, ParameterExpression)]
        if any(x_name in theta_names for x_name in x_names):
            raise AlgorithmError('Variational form and feature map are not allowed to share '
                                 'parameters with the same name!')

        qr = QuantumRegister(self._num_qubits, name='q')
        cr = ClassicalRegister(self._num_qubits, name='c')
        qc = QuantumCircuit(qr, cr)

        if isinstance(self.feature_map, QuantumCircuit):
            param_dict = dict(zip(self._feature_map_params, x))
            circuit = self._feature_map.assign_parameters(param_dict, inplace=False)
            qc.append(circuit.to_instruction(), qr)
        else:
            qc += self._feature_map.construct_circuit(x, qr)

        if isinstance(self.var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, theta))
            circuit = self._var_form.assign_parameters(param_dict, inplace=False)
            qc.append(circuit.to_instruction(), qr)
        else:
            qc += self._var_form.construct_circuit(theta, qr)

        if measurement:
            qc.barrier(qr)
            qc.measure(qr, cr)
        return qc

    def _get_prediction(self, data, theta):
        """Make prediction on data based on each theta.

        Args:
            data (numpy.ndarray): 2-D array, NxD, N data points, each with D dimension
            theta (list[numpy.ndarray]): list of 1-D array, parameters sets for variational form

        Returns:
            Union(numpy.ndarray or [numpy.ndarray], numpy.ndarray or [numpy.ndarray]):
                list of NxK array, list of Nx1 array
        """
        from qiskit.ml.circuit.library import RawFeatureVector
        circuits = []

        num_theta_sets = len(theta) // self._var_form.num_parameters
        theta_sets = np.split(theta, num_theta_sets)

        def _build_parameterized_circuits():
            var_form_support = isinstance(self._var_form, QuantumCircuit) \
                or self._var_form.support_parameterized_circuit
            feat_map_support = isinstance(self._feature_map, QuantumCircuit) \
                or self._feature_map.support_parameterized_circuit

            # cannot transpile the RawFeatureVector
            if isinstance(self._feature_map, RawFeatureVector):
                feat_map_support = False

            if var_form_support and feat_map_support and self._parameterized_circuits is None:
                parameterized_circuits = self.construct_circuit(
                    self._feature_map_params, self._var_form_params,
                    measurement=not self._quantum_instance.is_statevector)
                self._parameterized_circuits = \
                    self._quantum_instance.transpile(parameterized_circuits)[0]

        _build_parameterized_circuits()
        for thet in theta_sets:
            for datum in data:
                if self._parameterized_circuits is not None:
                    curr_params = dict(zip(self._feature_map_params, datum))
                    curr_params.update(dict(zip(self._var_form_params, thet)))
                    circuit = self._parameterized_circuits.assign_parameters(curr_params)
                else:
                    circuit = self.construct_circuit(
                        datum, thet, measurement=not self._quantum_instance.is_statevector)
                circuits.append(circuit)

        results = self._quantum_instance.execute(
            circuits, had_transpiled=self._parameterized_circuits is not None)

        circuit_id = 0
        predicted_probs = []
        predicted_labels = []
        for _ in theta_sets:
            counts = []
            for _ in data:
                if self._quantum_instance.is_statevector:
                    temp = results.get_statevector(circuit_id)
                    outcome_vector = (temp * temp.conj()).real
                    # convert outcome_vector to outcome_dict, where key
                    # is a basis state and value is the count.
                    # Note: the count can be scaled linearly, i.e.,
                    # it does not have to be an integer.
                    outcome_dict = {}
                    bitstr_size = int(math.log2(len(outcome_vector)))
                    for i, _ in enumerate(outcome_vector):
                        bitstr_i = format(i, '0' + str(bitstr_size) + 'b')
                        outcome_dict[bitstr_i] = outcome_vector[i]
                else:
                    outcome_dict = results.get_counts(circuit_id)

                counts.append(outcome_dict)
                circuit_id += 1

            probs = return_probabilities(counts, self._num_classes)
            predicted_probs.append(probs)
            predicted_labels.append(np.argmax(probs, axis=1))

        if len(predicted_probs) == 1:
            predicted_probs = predicted_probs[0]
        if len(predicted_labels) == 1:
            predicted_labels = predicted_labels[0]

        return predicted_probs, predicted_labels

    # Breaks data into minibatches. Labels are optional,
    # but will be broken into batches if included.
    def batch_data(self, data, labels=None, minibatch_size=-1):
        """Batch data

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is dimension.
            labels (numpy.ndarray): Nx1 array, N is number of data.
            minibatch_size (int): the size of each minibatch.
        """
        label_batches = None

        if 0 < minibatch_size < len(data):
            batch_size = min(minibatch_size, len(data))
            if labels is not None:
                label_batches = np.array_split(np.random.shuffle(labels), batch_size)

            batches = np.array_split(np.random.shuffle(data), batch_size)
        else:
            batches = np.asarray([data])
            label_batches = np.asarray([labels])
        return batches, label_batches

    def is_gradient_really_supported(self):
        """ returns is gradient really supported """
        return self.optimizer.is_gradient_supported and not self.optimizer.is_gradient_ignored

    def train(self, data, labels, quantum_instance=None, minibatch_size=-1):
        """Train the models, and save results.

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
            minibatch_size (int): the size of each minibatched accuracy evaluation
        """
        self._quantum_instance = \
            self._quantum_instance if quantum_instance is None else quantum_instance
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size
        self._batches, self._label_batches = self.batch_data(data, labels, minibatch_size)
        print(self._batches, self._label_batches)
        self._batch_index = 0

        if self.initial_point is None:
            self.initial_point = self.random.standard_normal(self._var_form.num_parameters)

        self._eval_count = 0

        grad_fn = None
        if minibatch_size > 0 and self.is_gradient_really_supported():  # we need some wrapper
            grad_fn = self._gradient_function_wrapper

        result = self.find_minimum(initial_point=self.initial_point,
                                   var_form=self.var_form,
                                   cost_fn=self.loss,
                                   optimizer=self.optimizer,
                                   gradient_fn=grad_fn)

        # TODO remove - mimics former VQAlgorithm result dict so it can be extended
        self._ret = {}
        self._ret['num_optimizer_evals'] = result.optimizer_evals
        self._ret['min_val'] = result.optimal_value
        self._ret['opt_params'] = result.optimal_point
        self._ret['eval_time'] = result.optimizer_time

        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        del self._batches
        del self._label_batches
        del self._batch_index

        self._ret['training_loss'] = self._ret['min_val']

    # temporary fix: this code should be unified with the gradient api in optimizer.py
    def _gradient_function_wrapper(self, theta):
        """Compute and return the gradient at the point theta.

        Args:
            theta (numpy.ndarray): 1-d array

        Returns:
            numpy.ndarray: 1-d array with the same shape as theta. The  gradient computed
        """
        epsilon = 1e-8
        f_orig = self.loss(theta)
        grad = np.zeros((len(theta),), float)
        for k, _ in enumerate(theta):
            theta[k] += epsilon
            f_new = self.loss(theta)
            grad[k] = (f_new - f_orig) / epsilon
            theta[k] -= epsilon  # recover to the center state
        if self.is_gradient_really_supported():
            self._batch_index += 1  # increment the batch after gradient callback
        return grad

    def loss(self, theta):
        """Compute the loss over the current batch of data samples.

        Args:
            theta (numpy.ndarray): the current point in the parameter space.

        Returns:
            Union(float, list): the cost for the current batch of samples.
        """
        batch_index = self._batch_index % len(self._batches)
        predicted_probs, _ = self._get_prediction(self._batches[batch_index], theta)
        total_cost = []
        if not isinstance(predicted_probs, list):
            predicted_probs = [predicted_probs]
        for i, _ in enumerate(predicted_probs):
            curr_cost = cost_estimate(predicted_probs[i], self._label_batches[batch_index])
            total_cost.append(curr_cost)
            if self._callback is not None:
                self._callback(
                    self._eval_count,
                    theta[i * self._var_form.num_parameters:(i + 1)
                          * self._var_form.num_parameters],
                    curr_cost,
                    self._batch_index
                )
            self._eval_count += 1

        if not self.is_gradient_really_supported():
            self._batch_index += 1  # increment the batch after eval callback

        logger.debug('Intermediate batch cost: %s', sum(total_cost))
        return total_cost if len(total_cost) > 1 else total_cost[0]

    def test(self, data, labels, quantum_instance=None, minibatch_size=-1, params=None):
        """Predict the labels for the data, and test against with ground truth labels.

        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is data dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
            minibatch_size (int): the size of each minibatched accuracy evaluation
            params (list): list of parameters to populate in the variational form

        Returns:
            float: classification accuracy
        """
        # minibatch size defaults to setting in instance variable if not set
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size

        batches, label_batches = self.batch_data(data, labels, minibatch_size)
        self.batch_num = 0
        if params is None:
            params = self.optimal_params
        total_cost = 0
        total_correct = 0
        total_samples = 0

        self._quantum_instance = \
            self._quantum_instance if quantum_instance is None else quantum_instance
        for batch, label_batch in zip(batches, label_batches):
            predicted_probs, _ = self._get_prediction(batch, params)
            total_cost += cost_estimate(predicted_probs, label_batch)
            total_correct += np.sum((np.argmax(predicted_probs, axis=1) == label_batch))
            total_samples += label_batch.shape[0]
            int_accuracy = \
                np.sum((np.argmax(predicted_probs, axis=1) == label_batch)) / label_batch.shape[0]
            logger.debug('Intermediate batch accuracy: {:.2f}%'.format(int_accuracy * 100.0))
        total_accuracy = total_correct / total_samples
        logger.info('Accuracy is {:.2f}%'.format(total_accuracy * 100.0))
        self._ret['testing_accuracy'] = total_accuracy
        self._ret['test_success_ratio'] = total_accuracy
        self._ret['testing_loss'] = total_cost / len(batches)
        return total_accuracy

    def predict(self, data, quantum_instance=None, minibatch_size=-1, params=None):
        """Predict the labels for the data.

        Args:
            data (numpy.ndarray): NxD array, N is number of data, D is data dimension
            quantum_instance (QuantumInstance): quantum backend with all setting
            minibatch_size (int): the size of each minibatched accuracy evaluation
            params (list): list of parameters to populate in the variational form

        Returns:
            list: for each data point, generates the predicted probability for each class
            list: for each data point, generates the predicted label (that with the highest prob)
        """

        # minibatch size defaults to setting in instance variable if not set
        minibatch_size = minibatch_size if minibatch_size > 0 else self._minibatch_size
        batches, _ = self.batch_data(data, None, minibatch_size)
        if params is None:
            params = self.optimal_params
        predicted_probs = None
        predicted_labels = None

        self._quantum_instance = \
            self._quantum_instance if quantum_instance is None else quantum_instance
        for i, batch in enumerate(batches):
            if len(batches) > 0:  # pylint: disable=len-as-condition
                logger.debug('Predicting batch %s', i)
            batch_probs, batch_labels = self._get_prediction(batch, params)
            if not predicted_probs and not predicted_labels:
                predicted_probs = batch_probs
                predicted_labels = batch_labels
            else:
                predicted_probs = np.concatenate((predicted_probs, batch_probs))
                predicted_labels = np.concatenate((predicted_labels, batch_labels))
        self._ret['predicted_probs'] = predicted_probs
        self._ret['predicted_labels'] = predicted_labels
        return predicted_probs, predicted_labels

    def _run(self):
        self.train(self._training_dataset[0], self._training_dataset[1])

        if self._test_dataset is not None:
            self.test(self._test_dataset[0], self._test_dataset[1])

        if self._datapoints is not None:
            _, predicted_labels = self.predict(self._datapoints)
            self._ret['predicted_classes'] = map_label_to_class_name(predicted_labels,
                                                                     self._label_to_class)
        self.cleanup_parameterized_circuits()
        return self._ret

    def get_optimal_cost(self):
        """ get optimal cost """
        if 'opt_params' not in self._ret:
            raise AlgorithmError("Cannot return optimal cost before running the "
                                 "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        """ get optimal circuit """
        if 'opt_params' not in self._ret:
            raise AlgorithmError("Cannot find optimal circuit before running "
                                 "the algorithm to find optimal params.")
        if isinstance(self._var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, self._ret['opt_params']))
            return self._var_form.assign_parameters(param_dict)
        return self._var_form.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        """ get optimal vector """
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AlgorithmError("Cannot find optimal vector before running "
                                 "the algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc, decimals=16)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def feature_map(self) -> Optional[Union[FeatureMap, QuantumCircuit]]:
        """Return the feature map."""
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: Union[FeatureMap, QuantumCircuit]):
        """Set the feature map.

        Also sets the number of qubits, the internally stored feature map parameters and,
        if the feature map is a circuit, the order of the parameters.
        """
        if isinstance(feature_map, QuantumCircuit):
            # patch the feature dimension to the circuit
            feature_map.feature_dimension = len(feature_map.parameters)

            # store the parameters
            self._num_qubits = feature_map.num_qubits
            self._feature_map_params = sorted(feature_map.parameters, key=lambda p: p.name)
            self._feature_map = feature_map
        elif isinstance(feature_map, FeatureMap):
            warnings.warn('The qiskit.aqua.components.feature_maps.RawFeatureVector object is '
                          'deprecated as of 0.9.0 and will be removed no earlier than 3 months '
                          'after the release. You can use the RawFeatureVector from '
                          'qiskit.ml.circuit.library instead.',
                          DeprecationWarning, stacklevel=2)

            self._num_qubits = feature_map.num_qubits
            self._feature_map_params = ParameterVector('x', length=feature_map.feature_dimension)
            self._feature_map = feature_map
        else:
            raise ValueError('Unsupported type {} of feature_map.'.format(type(feature_map)))

        if self._feature_map.feature_dimension == 0:
            warnings.warn('The feature map has no parameters that can be optimized to represent '
                          'the data. This will most likely cause the VQC to fail.')

    @property
    def optimal_params(self):
        """ returns optimal parameters """
        if 'opt_params' not in self._ret:
            raise AlgorithmError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']

    @property
    def ret(self):
        """ returns result """
        return self._ret

    @ret.setter
    def ret(self, new_value):
        """ sets result """
        self._ret = new_value

    @property
    def label_to_class(self):
        """ returns label to class """
        return self._label_to_class

    @property
    def class_to_label(self):
        """ returns class to label """
        return self._class_to_label

    def load_model(self, file_path):
        """ load model """
        model_npz = np.load(file_path, allow_pickle=True)  # pylint: disable=unexpected-keyword-arg
        self._ret['opt_params'] = model_npz['opt_params']

    def save_model(self, file_path):
        """ save model """
        model = {'opt_params': self._ret['opt_params']}
        np.savez(file_path, **model)

    @property
    def test_dataset(self):
        """ returns test dataset """
        return self._test_dataset

    @property
    def training_dataset(self):
        """ returns training dataset """
        return self._training_dataset

    @property
    def datapoints(self):
        """ return data points """
        return self._datapoints


def assign_label(measured_key, num_classes):
    """
    Classes = 2:
    - If odd number of qubits we use majority vote
    - If even number of qubits we use parity
    Classes = 3
    - We use part-parity
      {ex. for 2 qubits: [00], [01,10], [11] would be the three labels}

    Args:
        measured_key (str): measured key
        num_classes (int): number of classes
    Returns:
        int: key order
    """
    measured_key = np.asarray([int(k) for k in list(measured_key)])
    num_qubits = len(measured_key)
    if num_classes == 2:
        if num_qubits % 2 != 0:
            total = np.sum(measured_key)
            return 1 if total > num_qubits / 2 else 0
        else:
            hamming_weight = np.sum(measured_key)
            is_odd_parity = hamming_weight % 2
            return is_odd_parity

    elif num_classes == 3:
        first_half = int(np.floor(num_qubits / 2))
        modulo = num_qubits % 2
        # First half of key
        hamming_weight_1 = np.sum(measured_key[0:first_half + modulo])
        # Second half of key
        hamming_weight_2 = np.sum(measured_key[first_half + modulo:])
        is_odd_parity_1 = hamming_weight_1 % 2
        is_odd_parity_2 = hamming_weight_2 % 2

        return is_odd_parity_1 + is_odd_parity_2

    else:
        total_size = 2**num_qubits
        class_step = np.floor(total_size / num_classes)

        decimal_value = measured_key.dot(1 << np.arange(measured_key.shape[-1] - 1, -1, -1))
        key_order = int(decimal_value / class_step)
        return key_order if key_order < num_classes else num_classes - 1


def cost_estimate(probs, gt_labels, shots=None):  # pylint: disable=unused-argument
    """Calculate cross entropy.

    Args:
        shots (int): the number of shots used in quantum computing
        probs (numpy.ndarray): NxK array, N is the number of data and K is the number of class
        gt_labels (numpy.ndarray): Nx1 array

    Returns:
        float: cross entropy loss between estimated probs and gt_labels

    Note:
        shots is kept since it may be needed in future.
    """
    mylabels = np.zeros(probs.shape)
    for i in range(gt_labels.shape[0]):
        whichindex = gt_labels[i]
        mylabels[i][whichindex] = 1

    def cross_entropy(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        tmp = np.sum(targets * np.log(predictions), axis=1)
        ce = -np.sum(tmp) / N
        return ce

    x = cross_entropy(probs, mylabels)
    return x


def cost_estimate_sigmoid(shots, probs, gt_labels):
    """Calculate sigmoid cross entropy

    Args:
        shots (int): the number of shots used in quantum computing
        probs (numpy.ndarray): NxK array, N is the number of data and K is the number of class
        gt_labels (numpy.ndarray): Nx1 array

    Returns:
        float: sigmoid cross entropy loss between estimated probs and gt_labels
    """
    # Error in the order of parameters corrected below - 19 Dec 2018
    # x = cost_estimate(shots, probs, gt_labels)
    x = cost_estimate(probs, gt_labels, shots)
    loss = (1.) / (1. + np.exp(-x))
    return loss


def return_probabilities(counts, num_classes):
    """Return the probabilities of given measured counts

    Args:
        counts (list[dict]): N data and each with a dict recording the counts
        num_classes (int): number of classes

    Returns:
        numpy.ndarray: NxK array
    """

    probs = np.zeros(((len(counts), num_classes)))
    for idx, _ in enumerate(counts):
        count = counts[idx]
        shots = sum(count.values())
        for k, v in count.items():
            label = assign_label(k, num_classes)
            probs[idx][label] += v / shots
    return probs
