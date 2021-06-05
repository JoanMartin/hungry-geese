import abc

from game_state import GameState


class BaseEncoder(abc.ABC):

    @abc.abstractmethod
    def name(self):
        """
        Name of the encorder the model is using
        """
        pass

    @abc.abstractmethod
    def encode(self, game_state: GameState, goose_index: int):
        """
        Turns a Go board into numeric data
        """
        pass

    @abc.abstractmethod
    def num_points(self):
        """
        Number of points on the board: board width times board height
        """
        pass

    @abc.abstractmethod
    def shape(self):
        """
        Shape of the encoded board structure
        """
        pass
