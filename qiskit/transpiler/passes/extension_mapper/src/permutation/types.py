"""Type definitions used within the permutation package."""
from typing import TypeVar, Dict, Tuple, List

PermuteElement = TypeVar('PermuteElement')
Permutation = Dict[PermuteElement, PermuteElement]
Swap = Tuple[PermuteElement, PermuteElement]
CyclicPermutation = List[PermuteElement]
