from typing import Mapping, Optional, Iterator, List
from qiskit.transpiler.routing import Permutation, Swap


def permute_path(permutation: Permutation[int]) -> Iterator[List[Swap[int]]]: ...
def permute_path_partial(mapping: Mapping[int, int],
                             length: Optional[int] = None) -> Iterator[List[Swap[int]]]: ...
