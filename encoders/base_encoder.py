import abc

from kaggle_environments.envs.hungry_geese.hungry_geese import Action

from game_state import GameState


class BaseEncoder(abc.ABC):

    @abc.abstractmethod
    def name(self):
        """
        Name of the encoder the model is using
        """
        pass

    @abc.abstractmethod
    def encode(self, game_state: GameState, goose_index: int):
        """
        Turns a board into numeric data
        """
        pass

    @abc.abstractmethod
    def encode_action(self, action: Action):
        """
        Turns an action into an integer index
        """
        pass

    @abc.abstractmethod
    def decode_action_index(self, index: int):
        """
        Turns an integer index back into an action
        """
        pass

    @abc.abstractmethod
    def num_points(self):
        """
        Number of points on the board: board width times board height
        """
        pass

    @abc.abstractmethod
    def num_actions(self):
        """
        Number of actions
        """
        pass

    @abc.abstractmethod
    def shape(self):
        """
        Shape of the encoded board structure
        """
        pass
