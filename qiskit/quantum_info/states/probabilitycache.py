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
    Used to cache probability outcomes and StabilizerState when calculating the branches
    using targets. When using multiple targets for the Probability calculation, which will
    reduce the number of branches calculated, if a user has multiple branches they are
    calculating, this can save time on rebuilding the state of a branch that was partially
    traversed, and gives a better starting point and increases performance.
    """

    def __init__(self):
        self.cache_outcome: dict[str, float] = {}
        self.cache_ret: dict[str, QuantumState] = {}

    @staticmethod
    def cache_key(outcome: str | list[str]) -> str:
        """Calculate the cache key used for inserting and retrieving all cache entries
        if in str format it will be assumed this was calculated earlier

        Args:
            outcome (list[str]): outcome value used for building the key

        Returns:
            str: the key in str format
        """
        if isinstance(outcome, list):
            return "".join(outcome)
        else:
            return outcome

    def is_state_in_stabilizer_cache(self, outcome: str | list[str]) -> bool:
        """Check if the stabilizer state is cached from previous branch calculations

        Args:
            outcome list[str]: outcome value used to get the key and check the cache

        Returns:
            bool: True if state is in cache, False if not in cache
        """
        key: str = ProbabilityCache.cache_key(outcome)
        return ProbabilityCache._check_key(key) and (key in self.cache_ret)

    def retrieve_state(self, outcome: str | list[str]) -> QuantumState | None:
        """Retrieve the stabilizer state from the cache, if the state does not
        exist in the cache, None is returned

        Args:
            outcome list[str]: outcome value used to get the key and check the cache

        Returns:
            QuantumState | None: the state retrieved from the cache, None if key is
                not in in cache
        """
        try:
            return self.cache_ret[ProbabilityCache.cache_key(outcome)]
        except KeyError:
            return None

    def retrieve_outcome(self, outcome: str | list[str]) -> float | None:
        """Retrieve the outcome value based on the outcome value to build the key,
        if the state does not exist in the cache, None is returned

        Args:
            outcome list[str]: outcome value used to get the key and check the cache

        Returns:
            float | None: the cached float value for the outcome probability, None
                if key is not in cache
        """
        try:
            return self.cache_outcome[ProbabilityCache.cache_key(outcome)]
        except KeyError:
            return None

    def outcome_cache_contains_entries(self) -> bool:
        """Check if the outcome cache contains any entries

        Returns:
            bool: True if at least 1 entry exists, False if nothing is cached
        """
        return len(self.cache_outcome) > 0

    def insert_state(self, outcome: str | list[str], ret: QuantumState):
        """Insert into the state cache. Key is verified before inserting
        as it must NOT be a fully calculated branch, or a branch with no caclulations
        for example 'XXXX' or '00101' will not be cached.
        Must contain at least 1 'X' and 1 ('1' or '0') to cache

        Args:
            outcome (list[str]): outcome value used to get the key and check the cache
            ret (QuantumState): the QuantumState to save in the cache
        """
        key: str = ProbabilityCache.cache_key(outcome)
        if ProbabilityCache._check_key(key):
            self.cache_ret[key] = ret

    def insert_outcome(self, outcome: str | list[str], outcome_prob: float):
        """Insert into the outcome cache. Key is verified before inserting
        as it must NOT be a fully calculated branch, or a branch with no caclulations
        for example 'XXXX' or '00101' will not be cached.
        Must contain at least 1 'X' and 1 ('1' or '0') to cache

        Args:
            outcome list[str]: outcome value used to get the key and check the cache
            outcome_prob float: probability to save in the cache for the outcome
        """
        key: str = ProbabilityCache.cache_key(outcome)
        if ProbabilityCache._check_key(key):
            self.cache_outcome[key] = outcome_prob

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

    def retreive_key_for_most_completed_branch_to_target(self, target: str) -> str:
        """Retrieves the best starting point for calculating the probability
        if a partital branch was already calculated on the path of the target calculation
        It will retrieve the key for the closest node used to continue calculating the branch

        Args:
            cache dict[str, float]: cache value of previously calculated branch
            target str: target item wanting to calculate

        Returns:
        str: the key if it is found in the cache, or None if not found
        """
        key: list[str] = None
        test_key: str = None
        for level in range(1, len(target)):
            # Find the deepest branch calculated that can be used from the cache
            test_key = ("X" * level) + target[level:]
            if test_key in self.cache_outcome:
                # Must convert the key to a list as all the functions use the outcome
                # Variable for cache entries which is in this format
                key = test_key
                break
        return key
