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
    def check_precondition(input):
        """ If the preconditions are not met, the method will return False so
        the pipeline manager will jump the next stage, otherwise the Stage will
        be executed """
        pass

    @abstractmethod
    def handle_request(input):
        """ Here we will implement the logic of the stage. It receives an
        InputOutput object, which is a container where we will extract our
        input and will add the output for the next stages. At some point in the
        pipeline, we'll want to return a final output value. This ouput is
        going to be saved using the input.result property. It doesn't need to
        be in the last stage, this will be the most common use-case though. """
        pass
