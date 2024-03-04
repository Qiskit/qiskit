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
from qiskit.quantum_info.states.quantum_state import QuantumState

class ProbabilityCache:
    def __init__(self):
        self.cache_outcome: dict[str, float] = {}
        self.cache_ret: dict[str, QuantumState] = {}

    @staticmethod
    def cache_key(outcome: list[str]):
        return "".join(outcome)
    
    def is_outcome_in_cache(self, key: str) -> bool:
        return (ProbabilityCache._check_key(key) and key in self.cache_outcome)
    
    def is_state_in_stabilizer_cache(self, outcome: str) -> bool:
        key: str = ProbabilityCache.cache_key(outcome)
        return (ProbabilityCache._check_key(key) and (key in self.cache_ret))
    
    def retrieve_stabalizer_state(self, outcome: list[str]) -> QuantumState:
        return self.cache_ret[ProbabilityCache.cache_key(outcome)]
    
    def retrieve_outcome(self, outcome: list[str]) -> float:
        return self.cache_outcome[ProbabilityCache.cache_key(outcome)]
    
    def outcome_cache_contains_entries(self) -> bool:
        return (len(self.cache_outcome) > 0)
    
    def insert_stabalizer_state(self, outcome: list[str], ret: QuantumState):
        key: str = ProbabilityCache.cache_key(outcome)
        if(ProbabilityCache._check_key(key)):
            self.cache_ret[key] = ret

    def insert_outcome(self, outcome: list[str], outcome_prob: float):
        key: str = ProbabilityCache.cache_key(outcome)
        if(ProbabilityCache._check_key(key)):
            self.cache_outcome[key] = outcome_prob

    def _check_key(key: str) -> bool:
        return ('X' in key and ('1' in key or '0' in key))

    def retreive_best_cache_starting_point(self, target: str) -> list[str]:
        '''Helper to get the best starting point for calculating the probability 
        if a partital branch was already calculated

        Args:
            cache dict[str, float]: cache value of previously calculated branch
            target str: target item wanting to calculate

        Returns:
        str: the key if it is found in the cache, or None if not found
        '''
        key: list[str] = None
        test_key: str = None
        for level in range(1, len(target)):
            #Find the deepest branch calculated that can be used from the cache
            test_key = (('X' * level) + target[level:])
            if(test_key in self.cache_outcome):
                key = list(test_key)
                break 
        return key
