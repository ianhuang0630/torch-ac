from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel:
    recurrent = False
    optlib = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass


# a new type of model, OpLibModelBase(ACModel). This model is not recurrent.

class OpLibModelBase(ACModel):
    """
    The base class of all OptLibModels.
    This class maintains a library of options for every symbol within a dsl.
    """
    recurrent = False  # for now..
    optlib = True  # so that the code can differentiate this model type for special treatment

    @abstractmethod
    def forward(self, obs):
        pass

    # @abstractmethod
    # def set_task(self, task):
    #     """
    #     Setting the symbol sequence
    #     """

    # @abstractmethod
    # def reset(self):
    #     """
    #     Resetting all symbol sequences and indices.
    #     """
