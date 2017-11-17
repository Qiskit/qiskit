import logging

""" This class represents a container where all input and output data from the
stages will be added. Every stage is responsible of getting their required data
and adding their output at the end of the stage."""
class StageInputOutput(object):

    def __init__(self):
        self._data = {'result': None}

    def insert(self, key, value):
        """ Insert the data into the dictionary with the name as a key. If the
        name already exists, data will be replaced.
        Args:
            key (str): Name of the key. Format: name.of.the.key
            value : Data associated with the key
        """
        self._data[key] = value

    def get(self, key):
        """ Gets the data by the given name. If the data doesn't exist, will
        throw a StageError exception.
        Args:
            key (str): The key associated with the data to be returned
        Rises:
            StageError: If there's no such name in the dictionary"""
        try:
            return self._data[key]
        except Exception as ex:
            raise StageError('Could not retrieve key: {}'.format(key)) from ex

    @property
    def result(self):
        """ Commodity to get the final result of the pipeline execution.
        """
        return self._data['result']

    @result.setter
    def result(self, value):
        """ Commodity to set the final result of the pipeline execution.
        """
        self._data['result'] = value


    def exists(self, key):
        if not key in self._data:
            return False
        return True


