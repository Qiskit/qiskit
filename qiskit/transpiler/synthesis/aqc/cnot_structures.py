# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
These are the CNOT structure methods: anything that you need for CNOTS.
"""

import numpy as np

# N O T E: we use 1-based indices like in Matlab.

_NETWORK_LAYOUTS = {"sequ", "spin", "cart", "cyclic_spin", "cyclic_line"}
_CONNECTIVITY_TYPES = {"full", "line", "star"}


def check_cnots(nqubits: int, cnots: np.ndarray) -> bool:
    """
    Checks validity of CNOt structure.
    """
    assert isinstance(nqubits, (int, np.int64)) and nqubits >= 1
    assert isinstance(cnots, np.ndarray)
    assert cnots.dtype == np.int64 or cnots.dtype == int
    assert cnots.shape == (2, cnots.size // 2)
    # pylint: disable=misplaced-comparison-constant
    assert np.all(np.logical_and(1 <= cnots, cnots <= nqubits))  # 1-based!!
    assert np.all(cnots[0, :] != cnots[1, :])
    return True


def get_network_layouts() -> list:
    """Returns the list of supported network geometry types."""
    return list(_NETWORK_LAYOUTS)


def get_connectivity_types() -> list:
    """Returns the list of supported inter-qubit connectivity types."""
    return list(_CONNECTIVITY_TYPES)


def lower_limit(num_qubits: int) -> int:
    """
    Returns lower limit on the number of CNOT units that guarantees
    exact representation of a unitary operator by quantum gates.

    Args:
        num_qubits: number of qubits.

    Returns:
        lower limit on the number of CNOT units.
    """
    assert isinstance(num_qubits, (np.int64, int)) and num_qubits >= 2
    num_cnots = round(np.ceil((4 ** num_qubits - 3 * num_qubits - 1) / 4.0))
    return num_cnots


def make_cnot_network(
    num_qubits: int,
    network_layout: str = "spin",
    connectivity_type: str = "full",
    depth: int = 0,
    verbose: int = 0,
) -> np.ndarray:
    """
    Generates a network consisting of building blocks each containing a CNOT
    gate and possibly some single-qubit ones. This network models a quantum
    operator in question.
    Note, each building block has 2 input and outputs corresponding to a pair
    of qubits. What we actually return here is a chain of indices of qubit
    pairs shared by every building block in a row.

    Args:
        num_qubits: number of qubits.
        network_layout: type of network geometry, {"sequ", "spin", "cart"}.
        connectivity_type: type of inter-qubit connectivity,
                              {"full", "line", "star"}.
        depth: depth of the CNOT-network, i.e. the number of layers,
                  where each layer consists of a single CNOT-block; default
                  value will be selected, if L <= 0.
        verbose: verbosity level.

    Returns:
        2xN matrix that defines layers in cnot-network, where N is either
             equal L, or defined by a concrete type of the network.

    Raises:
         ValueError: if unsupported type of CNOT-network layout is passed.
    """
    assert isinstance(num_qubits, (np.int64, int)) and num_qubits >= 2
    assert isinstance(network_layout, str)
    assert isinstance(connectivity_type, str)
    assert isinstance(depth, (np.int64, int))
    assert isinstance(verbose, (np.int64, int))

    if depth <= 0:
        depth = lower_limit(num_qubits)
        if verbose > 0:
            print("#CNOT units chosen as the lower limit:", depth)

    if network_layout == "sequ":

        links = get_connectivity(num_qubits=num_qubits, connectivity=connectivity_type)
        return _sequential_network(num_qubits=num_qubits, links=links, depth=depth)

    elif network_layout == "spin":

        return _spin_network(num_qubits=num_qubits, links=dict(), depth=depth)

    elif network_layout == "cart":

        cnots = _cartan_network(num_qubits=num_qubits)
        if verbose > 0:
            print(
                "Optimal lower bound: ", lower_limit(num_qubits), "; Cartan CNOTs: ", cnots.shape[1]
            )
        return cnots

    elif network_layout == "cyclic_spin":

        assert connectivity_type == "full", "'{:s}' layout expects 'full' connectivity".format(
            network_layout
        )
        cnots = np.full((2, depth), fill_value=0, dtype=np.int64)
        z = 0
        while True:
            for i in range(0, num_qubits, 2):
                if i + 1 <= num_qubits - 1:
                    cnots[0, z] = i
                    cnots[1, z] = i + 1
                    z += 1
                if z >= depth:
                    cnots += 1  # 1-based index
                    return cnots

            for i in range(1, num_qubits, 2):
                if i + 1 <= num_qubits - 1:
                    cnots[0, z] = i
                    cnots[1, z] = i + 1
                    z += 1
                elif i == num_qubits - 1:
                    cnots[0, z] = i
                    cnots[1, z] = 0
                    z += 1
                if z >= depth:
                    cnots += 1  # 1-based index
                    return cnots

    elif network_layout == "cyclic_line":

        assert connectivity_type == "line", "'{:s}' layout expects 'line' connectivity".format(
            network_layout
        )
        cnots = np.full((2, depth), fill_value=0, dtype=np.int64)
        for i in range(depth):
            cnots[0, i] = (i + 0) % num_qubits
            cnots[1, i] = (i + 1) % num_qubits
        cnots += 1  # 1-based index
        return cnots

    else:
        raise ValueError(
            "unknown type of CNOT-network layout, "
            "expects one of {}, got {:s}".format(get_network_layouts(), network_layout)
        )


def generate_random_cnots(num_qubits: int, depth: int, set_depth_limit: bool = True) -> np.ndarray:
    """
    Generates a random CNot network for debugging and testing.
    N O T E: 1-based index.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network; it will be bounded by the lower limit
                  (function lower_limit()), if exceeded.
        set_depth_limit: apply theoretical upper limit on cnot circuit depth.

    Returns:
        2-x-depth matrix where each column contains two distinct indices
             from the range [1 ... nqubits];
    """
    assert isinstance(num_qubits, (np.int64, int)) and 1 <= num_qubits <= 16
    # pylint: disable=misplaced-comparison-constant
    assert isinstance(depth, (np.int64, int)) and 1 <= depth
    assert isinstance(set_depth_limit, bool)

    if set_depth_limit:
        depth = min(depth, lower_limit(num_qubits))
    cnots = np.tile(np.arange(num_qubits).reshape(num_qubits, 1), depth)
    for i in range(depth):
        np.random.shuffle(cnots[:, i])
    cnots = cnots[0:2, :].copy()
    cnots += 1  # 1-based index
    return cnots


def get_connectivity(num_qubits: int, connectivity: str) -> dict:
    """
    Generates connectivity structure between qubits.

    Args:
        num_qubits: number of qubits.
        connectivity: type of connectivity structure, {"full", "line", "star"}.

    Returns:
        dictionary of allowed links between qubits.

    Raises:
         ValueError: if unsupported type of CNOT-network layout is passed.
    """
    assert isinstance(num_qubits, (np.int64, int)) and num_qubits > 0
    assert isinstance(connectivity, str)

    if num_qubits == 1:

        links = {1: [1]}

    elif connectivity == "full":

        # Full connectivity between qubits.
        links = {i + 1: [j + 1 for j in range(num_qubits)] for i in range(num_qubits)}

    elif connectivity == "line":

        # Every qubit is connected to its immediate neighbours only.
        links = {i + 1: [i, i + 1, i + 2] for i in range(1, num_qubits - 1)}
        links[1] = [1, 2]
        links[num_qubits] = [num_qubits - 1, num_qubits]

    elif connectivity == "star":

        # Every qubit is connected to the first one only.
        links = {i + 1: [1, i + 1] for i in range(1, num_qubits)}
        links[1] = [j + 1 for j in range(num_qubits)]

    else:
        raise ValueError(
            "unknown connectivity type, expects one of {}, got {:s}".format(
                get_connectivity_types(), connectivity
            )
        )
    return links


def _sequential_network(num_qubits: int, links: dict, depth: int = 0) -> np.ndarray:
    """
    TODO: description

    Args:
        num_qubits: number of qubits.
        links: dictionary of connectivity links.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        2xL matrix that defines layers in qubit network.
    """
    assert len(links) > 0 and depth > 0
    l = 0
    A = np.full((2, depth), fill_value=0, dtype=np.int64)
    while True:
        for i in range(1, num_qubits):
            for j in range(i + 1, num_qubits + 1):
                if j in links[i]:
                    A[0, l] = i
                    A[1, l] = j
                    l += 1
                    if l >= depth:
                        return A


def _spin_network(num_qubits: int, links: dict, depth: int = 0) -> np.ndarray:
    """
    TODO: links is not used, why???
    TODO: description

    Args:
        num_qubits: number of qubits.
        links: dictionary of connectivity links.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        2xL matrix that defines layers in qubit network.
    """
    assert isinstance(links, dict) and depth > 0  # and len(links) > 0
    l = 0
    A = np.full((2, depth), fill_value=0, dtype=np.int64)
    while True:
        for i in range(1, num_qubits, 2):  # <--- starts from 1
            A[0, l] = i
            A[1, l] = i + 1
            l += 1
            if l >= depth:
                return A

        for i in range(2, num_qubits, 2):  # <--- starts from 2
            A[0, l] = i
            A[1, l] = i + 1
            l += 1
            if l >= depth:
                return A


def _cartan_network(num_qubits: int) -> np.ndarray:
    """
    TODO: description
    Cartan decomposition in a recursive way, starting from n = 3.

    Args:
        num_qubits: number of qubits.

    Returns:
        2xN matrix that defines layers in qubit network, where N is the
             depth of Cartan decomposition.

    Raises:
        ValueError: if number of qubits is less than 3.
    """
    n = num_qubits
    if n > 3:
        cnots = np.array([[1, 1, 1], [2, 2, 2]])
        mult = np.array([[n - 1, n - 2, n - 1, n - 2], [n, n, n, n]])
        for _ in range(n - 2):
            cnots = np.hstack((np.tile(np.hstack((cnots, mult)), 3), cnots))
            mult[0, -1] -= 1
            mult = np.tile(mult, 2)
    elif n == 3:
        cnots = np.array(
            [
                [1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1],
                [2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2],
            ]
        )
    else:
        raise ValueError(f"The number of qubits must be >= 3, got {n}.")

    return cnots
