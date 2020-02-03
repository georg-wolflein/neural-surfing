from abc import ABC, abstractmethod

from problems import Problem


class Agent(ABC):

    def __init__(self, problem: Problem):
        self.problem = problem

    @abstractmethod
    def train(self, epochs: int):
        pass
