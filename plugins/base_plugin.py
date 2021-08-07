from abc import ABC, abstractmethod


class BasePlugin(ABC):
    def __init__(self, print_when_finish):
        self.finish = False
        self.print_when_finish = print_when_finish

    @abstractmethod
    def execute(self, frame):
        pass
