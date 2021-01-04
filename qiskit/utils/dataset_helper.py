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

""" Data set helper """

import operator
from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA


def get_num_classes(dataset):
    """
    Check number of classes in a given dataset

    Args:
        dataset(dict): key is the class name and value is the data.

    Returns:
        int: number of classes
    """
    return len(list(dataset.keys()))


def get_feature_dimension(dataset):
    """
    Check feature dimension of a given dataset

    Args:
        dataset(dict): key is the class name and value is the data.

    Returns:
        int: feature dimension, -1 denotes no data in the dataset.
    Raises:
        TypeError: invalid data set
    """
    if not isinstance(dataset, dict):
        raise TypeError("Dataset is not formatted as a dict. Please check it.")

    feature_dim = -1
    for v in dataset.values():
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        return v.shape[1]

    return feature_dim


# pylint: disable=invalid-name
def split_dataset_to_data_and_labels(dataset, class_names=None):
    """
    Split dataset to data and labels numpy array

    If `class_names` is given, use the desired label to class name mapping,
    or create the mapping based on the keys in the dataset.

    Args:
        dataset (dict): {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
        class_names (dict): class name of dataset, {class_name: label}

    Returns:
        Union(tuple(list, dict), list):
            List contains two arrays of numpy.ndarray type
            where the array at index 0 is data, an NxD array, and at
            index 1 it is labels, an Nx1 array, containing values in range
            0 to K-1, where K is the number of classes. The dict is a map
            {str: int}, mapping class name to label. The tuple of list, dict is returned
            when `class_names` is not None, otherwise just the list is returned.

    Raises:
        KeyError: data set invalid
    """
    data = []
    labels = []
    if class_names is None:
        sorted_classes_name = sorted(list(dataset.keys()))
        class_to_label = {k: idx for idx, k in enumerate(sorted_classes_name)}
    else:
        class_to_label = class_names
    sorted_label = sorted(class_to_label.items(), key=operator.itemgetter(1))
    for class_name, _ in sorted_label:
        values = dataset[class_name]
        for value in values:
            data.append(value)
            try:
                labels.append(class_to_label[class_name])
            except Exception as ex:  # pylint: disable=broad-except
                raise KeyError('The dataset has different class names to '
                               'the training data. error message: {}'.format(ex)) from ex
    data = np.asarray(data)
    labels = np.asarray(labels)
    if class_names is None:
        return [data, labels], class_to_label
    else:
        return [data, labels]


def map_label_to_class_name(predicted_labels, label_to_class):
    """
    Helper converts labels (numeric) to class name (string)

    Args:
        predicted_labels (numpy.ndarray): Nx1 array
        label_to_class (dict or list): a mapping form label (numeric) to class name (str)

    Returns:
        str: predicted class names of each datum
    """

    if not isinstance(predicted_labels, np.ndarray):
        predicted_labels = np.asarray([predicted_labels])

    predicted_class_names = []

    for predicted_label in predicted_labels:
        predicted_class_names.append(label_to_class[predicted_label])
    return predicted_class_names


def reduce_dim_to_via_pca(x, dim):
    """
    Reduce the data dimension via pca

    Args:
        x (numpy.ndarray): NxD array
        dim (int): the targeted dimension D'

    Returns:
        numpy.ndarray: NxD' array
    """
    x_reduced = PCA(n_components=dim).fit_transform(x)
    return x_reduced


def discretize_and_truncate(data, bounds, num_qubits, return_data_grid_elements=False,
                            return_prob=False, prob_non_zero=True):
    """
    Discretize & truncate classical data to enable digital encoding in qubit registers
    whereby the data grid is [[grid elements dim 0],..., [grid elements dim k]]

    Args:
        data (list or array or np.array): training data (int or float) of dimension k
        bounds (list or array or np.array):  k min/max data values
            [[min_0,max_0],...,[min_k-1,max_k-1]] if univariate data: [min_0,max_0]
        num_qubits (list or array or np.array): k numbers of qubits to determine
            representation resolution, i.e. n qubits enable the representation of 2**n
            values [num_qubits_0,..., num_qubits_k-1]
        return_data_grid_elements (Bool): if True - return an array with the data grid
            elements
        return_prob (Bool): if True - return a normalized frequency count of the discretized and
            truncated data samples
        prob_non_zero (Bool): if True - set 0 values in the prob_data to 10^-1 to avoid potential
            problems when using the probabilities in loss functions - division by 0

    Returns:
        array: discretized and truncated data
        array: data grid [[grid elements dim 0],..., [grid elements dim k]]
        array: grid elements, Product_j=0^k-1 2**num_qubits_j element vectors
        array: data probability, normalized frequency count sorted from smallest to biggest element

    """
    # Truncate the data
    if np.ndim(bounds) == 1:
        bounds = np.reshape(bounds, (1, len(bounds)))

    data = data.reshape((len(data), len(num_qubits)))
    temp = []
    for i, data_sample in enumerate(data):
        append = True
        for j, entry in enumerate(data_sample):
            if entry < bounds[j, 0]:
                append = False
            if entry > bounds[j, 1]:
                append = False
        if append:
            temp.append(list(data_sample))
    data = np.array(temp)

    # Fit the data to the data element grid
    for j, prec in enumerate(num_qubits):
        data_row = data[:, j]  # dim j of all data samples
        # prepare element grid for dim j
        elements_current_dim = np.linspace(bounds[j, 0], bounds[j, 1], (2 ** prec))
        # find index for data sample in grid
        index_grid = np.searchsorted(
            elements_current_dim,
            data_row - (elements_current_dim[1] - elements_current_dim[0]) * 0.5)
        for k, index in enumerate(index_grid):
            data[k, j] = elements_current_dim[index]
        if j == 0:
            if len(num_qubits) > 1:
                data_grid = [elements_current_dim]
            else:
                data_grid = elements_current_dim
            grid_elements = elements_current_dim
        elif j == 1:
            temp = []
            for grid_element in grid_elements:
                for element_current in elements_current_dim:
                    temp.append([grid_element, element_current])
            grid_elements = temp
            data_grid.append(elements_current_dim)
        else:
            temp = []
            for grid_element in grid_elements:
                for element_current in elements_current_dim:
                    temp.append(deepcopy(grid_element).append(element_current))
            grid_elements = deepcopy(temp)
            data_grid.append(elements_current_dim)
    data_grid = np.array(data_grid)

    data = np.reshape(data, (len(data), len(data[0])))

    if return_prob:
        if np.ndim(data) > 1:
            prob_data = np.zeros(int(np.prod(np.power(np.ones(len(data[0])) * 2, num_qubits))))
        else:
            prob_data = np.zeros(int(np.prod(np.power(np.array([2]), num_qubits))))
        for data_element in data:
            for i, element in enumerate(grid_elements):
                if all(data_element == element):
                    prob_data[i] += 1 / len(data)
        if prob_non_zero:
            # add epsilon to avoid 0 entries which can be problematic in loss functions (division)
            prob_data = [1e-10 if x == 0 else x for x in prob_data]

        if return_data_grid_elements:
            return data, data_grid, grid_elements, prob_data
        else:
            return data, data_grid, prob_data

    else:
        if return_data_grid_elements:
            return data, data_grid, grid_elements

        else:
            return data, data_grid
