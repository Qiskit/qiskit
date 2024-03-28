# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
""" Probabilities Cache class. """

from __future__ import annotations
from qiskit.quantum_info.states.quantum_state import QuantumState


class ProbabilityCache:
    """
    Used to cache probability outcomes and QuantumState when measuring qubits
    using targets. When using multiple targets for the Probability calculations,
    this will reduce the number of repetative branches calculated. If a user has
    multiple targets they are measuring, this can save time on rebuilding the
    QuantumState and probability measurements and gives a better
    starting point to increase performance.
    """

    def __init__(self):
        self.cache_outcome: dict[str, float] = {}
        self.cache_ret: dict[str, QuantumState] = {}

    @staticmethod
    def _cache_key(outcome: str | list[str]) -> str:
        """Calculate the cache key used for inserting and retrieving all cache entries
        if in str format it will be assumed this was calculated earlier

        Args:
            outcome (str or list[str]): outcome value used build the key and check the cache

        Returns:
            str: the key in str format
        """
        if isinstance(outcome, list):
            return "".join(outcome)
        else:
            return outcome

    def is_state_cached(self, outcome: str | list[str]) -> bool:
        """Check if the QuantumState is cached from previous branch calculations

        Args:
            outcome (str or list[str]): outcome used to build the key and check the cache

        Returns:
            bool: True if state is in cache, False if not in cache
        """
        key: str = ProbabilityCache._cache_key(outcome)
        return ProbabilityCache._check_key(key) and (key in self.cache_ret)

    def retrieve_state(self, outcome: str | list[str]) -> QuantumState | None:
        """Retrieve the QuantumState from the cache, if the state does not
        exist in the cache, None is returned

        Args:
            outcome (str or list[str]): outcome used to build the key and check the cache

        Returns:
            (QuantumState or None): the state retrieved from the cache, None if key is
                not in in cache
        """
        try:
            return self.cache_ret[ProbabilityCache._cache_key(outcome)]
        except KeyError:
            return None

    def retrieve_outcome(self, outcome: str | list[str]) -> float | None:
        """Retrieve the outcome value based on the outcome to build the key,
        if the state does not exist in the cache, None is returned

        Args:
            outcome (str or list[str]): outcome used to build the key and check the cache

        Returns:
            (float or None): the cached float value for the outcome probability, None
                if key is not in cache
        """
        try:
            return self.cache_outcome[ProbabilityCache._cache_key(outcome)]
        except KeyError:
            return None

    def contains_entries(self) -> bool:
        """Check if the outcome cache contains any entries

        Returns:
            bool: True if at least 1 entry exists, False if nothing is cached
        """
        return len(self.cache_outcome) > 0 and len(self.cache_ret) > 0

    def insert(self, outcome: str | list[str], outcome_prob: float, ret: QuantumState) -> None:
        """Insert into cache the outcome and state. Key is verified before inserting as
        it must NOT be a fully calculated branch, or a branch with no measurements for
        example 'XXXX' or '00101' will not be cached.
        Must contain at least 1 'X' and 1 ('1' or '0') to cache

        Args:
            outcome (str or list[str]): outcome used to build the key and check the cache
            outcome_prob (float): probability to save in the cache for the outcome
            ret (QuantumState): the QuantumState to save in the cache
        """
        key: str = ProbabilityCache._cache_key(outcome)
        if ProbabilityCache._check_key(key):
            self.cache_outcome[key] = outcome_prob
            self.cache_ret[key] = ret

    @staticmethod
    def _check_key(key: str) -> bool:
        """Verify the key is valid for inserting into the cache
        Key must NOT be a fully calculated branch, or a branch with no caclulations
        for example 'XXXX' or '00101' will not be cached.
        Must contain at least 1 'X' and 1 ('1' or '0') to cache

        Args:
            key (str): the key to verify

        Returns:
            bool: True if it meets the criteria
        """
        return key is not None and ("X" in key) and ("1" in key or "0" in key)

    def retrieve_closest_outcome(self, target: str) -> list[str] | None:
        """Retrieves the best starting point for calculating the probability
        if a partital branch was already calculated on the path of the target calculation
        It will retrieve the key for the closest node used to continue calculating the branch

        Args:
            target (str): target item wanting to calculate

        Returns:
            list[str] | None: the outcome of of the most completed cached entry, or None if not found
        """
        key: list[str] = None
        test_key: str = None
        for level in range(1, len(target)):
            # Find the deepest branch calculated that can be used from the cache
            test_key = ("X" * level) + target[level:]
            if test_key in self.cache_outcome:
                # Must convert the key to a list as all the functions use the outcome
                # Variable for cache entries which is in this format
                key = list(test_key)
                break
        return key
