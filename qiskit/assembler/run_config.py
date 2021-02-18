# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Models for RunConfig and its related components."""

from types import SimpleNamespace


class RunConfig(SimpleNamespace):
    """Class for Run Configuration.

    Attributes:
        shots (int): the number of shots
        max_credits (int): the max_credits to use on the IBM Q public devices
        seed_simulator (int): the seed to use in the simulator
        memory (bool): whether to request memory from backend (per-shot
            readouts)
        parameter_binds (list[dict]): List of parameter bindings
    """

    def __init__(
        self,
        shots=None,
        max_credits=None,
        seed_simulator=None,
        memory=None,
        parameter_binds=None,
        **kwargs,
    ):
        """Initialize a RunConfig object

        Args:
            shots (int): the number of shots
            max_credits (int): the max_credits to use on the IBM Q public
                devices
            seed_simulator (int): the seed to use in the simulator
            memory (bool): whether to request memory from backend
                (per-shot readouts)
            parameter_binds (list[dict]): List of parameter bindings
            **kwargs: optional fields
        """
        if shots is not None:
            self.shots = shots
        if max_credits is not None:
            self.max_credits = max_credits
        if seed_simulator is not None:
            self.seed_simulator = seed_simulator
        if memory is not None:
            self.memory = memory
        if parameter_binds is not None:
            self.parameter_binds = parameter_binds
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data):
        """Create a new RunConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the RunConfig to create.
                         It will be in the same format as output by
                         :meth:`to_dict`.

        Returns:
            RunConfig: The RunConfig from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the RunConfig

        Returns:
            dict: The dictionary form of the RunConfig.
        """
        return self.__dict__
