"""
application generator
"""
from application.qft import QFT
from application.quantum_volume import QuantumVolume
from application.counterfeit_coin import CounterfeitCoin
from application.berstein_vazirani import BersteinVazirani


class ApplicationGenerator:
    """
    application generator
    """
    def __init__(self, seed):
        self.application_list = [QFT(seed), QuantumVolume(seed),
                                 CounterfeitCoin(seed), BersteinVazirani(seed)]
        return

    def get_app(self, name):
        """
        get application generator
        """
        for app in self.application_list:
            if name == app.name:
                return app
        return None

    def get_app_name_list(self):
        """
        get application list
        """
        name_list = []
        for app in self.application_list:
            name_list.append(app.name)
        return name_list
