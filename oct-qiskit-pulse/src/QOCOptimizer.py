from abc import ABC, abstractmethod


class QOCOptimizer(ABC):
    def get_pulse_schedule(self, gate, targets):
        pass

