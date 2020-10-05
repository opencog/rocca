import abc

class Wrapper(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def restart_decorator(restart):
        pass

    @staticmethod
    @abc.abstractmethod
    def step_decorator(step):
        pass

    @abc.abstractmethod
    def restart(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass
