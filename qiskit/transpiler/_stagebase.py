from abc import ABC, abstractmethod

""" This module implements the abstract base class for Stages of the transpiler
pipline.

To create a new Stage, subclass the StageBase class and register it with the
Pipeline object manager.

TODO: Explain declarative way to create new pipelines"""
class StageBase(ABC):
    @abstractmethod
    def __init__(self):
        self.name = None
        self.next_stage = None

    @abstractmethod
    def get_name(name):
        """ Gets a unique name for this stage. There cannot be two stages with
        the same name. """
        # TODO: Maybe, set a default random name?
        pass

    @abstractmethod
    def handle_request(input):
        """ Here we will implement the logic of the stage. It receives an
        InputOutput object, which is a container where we will extract our
        input and will add the output for the next stages. If we are the last
        stage (self.next_stage = None) then we won't add anything to the
        InputOutput object, and will return the raw value"""
        pass
