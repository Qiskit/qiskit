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


"""Tests for ProbabilityCache quantum state class."""
from __future__ import annotations
import itertools
import random
import unittest
import logging
from ddt import ddt
import numpy as np
from qiskit.quantum_info.states.probabilitycache import ProbabilityCache
from test import combine  # pylint: disable=wrong-import-order
from test import QiskitTestCase  # pylint: disable=wrong-import-order

logger = logging.getLogger(__name__)


@ddt
class TestProbabilityCache(QiskitTestCase):
    """Tests for StabilizerState class."""

    @combine(num_qubits=[2, 3, 4, 5, 7, 9, 12, 13])
    def test_cache_store_and_retrieve(self, num_qubits: int):
        """Partially fills the cache, checks that values exist
        in cache, and non-inserted values do not, then fill
        the cache with the rest of the values, and verify all
        the values exist in the cache and retrieve the correct values

        Args:
            num_qubits int: number of qubits to use for testing the cache
        """
        cache: ProbabilityCache = ProbabilityCache()
        # Build all combinations for 'X, 0, 1' possible combinations to store
        test_input: list[str] = [
            "".join(map(str, i)) for i in itertools.product(["X", "0", "1"], repeat=num_qubits)
        ]

        # Cache half of the items
        test_input_to_cache: list[str] = random.sample(test_input, int(len(test_input) / 2))

        # Build probabilties for all items to be cached
        dict_random_probs_to_insert: dict[str, float] = {
            item: random.uniform(0, 1) for item in test_input_to_cache
        }
        list_not_cached: list[str] = list(np.setdiff1d(test_input, test_input_to_cache))

        # Verify cache is empty
        self.assertFalse(cache.outcome_cache_contains_entries())
        # Add items to cache
        for i, key in enumerate(dict_random_probs_to_insert):
            cache.insert_outcome(key, dict_random_probs_to_insert[key])

        # Add items to cache for state, use int
        for i, key in enumerate(dict_random_probs_to_insert, 0):
            # Switch between passing the key as a list[str] and a str
            cache.insert_state(self._key_type(key, self._odd_num(i)), i)

        # Verfiy cache has at least 1 entry
        self.assertTrue(cache.outcome_cache_contains_entries())

        # Verify values in cache exist and are the correct value
        for i, key in enumerate(dict_random_probs_to_insert):
            key_t = self._key_type(key, not self._odd_num(i))
            # formula to decide if key was allowed to be inserted in cache
            if cache._check_key(key):
                self.assertTrue(cache.retrieve_outcome(key_t) == dict_random_probs_to_insert[key])
                self.assertTrue(cache.is_state_in_quantum_state_cache(key_t))
                self.assertTrue(cache.retrieve_state(key_t) == i)
            else:
                self.assertTrue(cache.retrieve_outcome(key_t) is None)
                self.assertFalse(cache.is_state_in_quantum_state_cache(key_t))
                self.assertTrue(cache.retrieve_state(key_t) is None)

        # Verify all non inserted items not in cache
        for i, key in enumerate(list_not_cached):
            key_t = self._key_type(key, not self._odd_num(i))
            self.assertTrue(cache.retrieve_outcome(key_t) is None)
            self.assertTrue(cache.retrieve_state(key_t) is None)

        items_to_cache_next: dict[str, float] = {
            item: random.uniform(0, 1) for item in list_not_cached
        }

        # Add items to cache
        for i, key in enumerate(items_to_cache_next):
            cache.insert_outcome(
                self._key_type(key, not self._odd_num(i)), items_to_cache_next[key]
            )

        # Add items to cache for state, use int
        for i, key in enumerate(items_to_cache_next, len(dict_random_probs_to_insert)):
            cache.insert_state(self._key_type(key, not self._odd_num(i)), i)

        # Verify values in cache exist and are the correct value
        for i, key in enumerate(dict_random_probs_to_insert):
            key_t = self._key_type(key, self._odd_num(i))
            # formula to decide if key was allowed to be inserted in cache
            if cache._check_key(key):
                self.assertTrue(cache.retrieve_outcome(key_t) == dict_random_probs_to_insert[key])
                self.assertTrue(cache.is_state_in_quantum_state_cache(key_t))
                self.assertTrue(cache.retrieve_state(key_t) == i)
            else:
                self.assertTrue(cache.retrieve_outcome(key_t) is None)
                self.assertFalse(cache.is_state_in_quantum_state_cache(key_t))
                self.assertTrue(cache.retrieve_state(key_t) is None)

        # Verify values in cache exist and are the correct value
        for i, key in enumerate(items_to_cache_next, len(dict_random_probs_to_insert)):
            # formula to decide if key was allowed to be inserted in cache
            key_t = self._key_type(key, self._odd_num(i))
            if cache._check_key(key):
                self.assertTrue(cache.retrieve_outcome(key_t) == items_to_cache_next[key])
                self.assertTrue(cache.is_state_in_quantum_state_cache(key_t))
                self.assertTrue(cache.retrieve_state(key_t) == i)
            else:
                self.assertTrue(cache.retrieve_outcome(key_t) is None)
                self.assertFalse(cache.is_state_in_quantum_state_cache(key_t))
                self.assertTrue(cache.retrieve_state(key_t) is None)

        keys_to_pick: list[str] = [
            key for key in test_input_to_cache if (key.count("X") == 1 and key[0] == "X")
        ]
        if len(keys_to_pick) > 0:
            # Only verify cache key where a valid starting place can be found
            choice: str = random.choice(keys_to_pick)
            choice_find: str = choice.replace("X", "1")
            self.assertTrue(
                cache.retreive_key_for_most_completed_branch_to_target(choice_find) == choice
            )

    @staticmethod
    def _key_type(key: str, as_list: bool):
        """Switch between passing the key as a list[str] and a str

        Returns:
            str | list[str]: key in form
        """
        return list(key) if as_list else key

    @staticmethod
    def _odd_num(val: int) -> bool:
        return val % 2


if __name__ == "__main__":
    unittest.main()
