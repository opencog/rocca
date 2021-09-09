import abc


class Wrapper(abc.ABC):
    """
    This wrapper abstracts and decorates components of environments in order to
    run agents on any environment. It is supposed to be derived by concrete
    environment wrappers with the following specifications.

    1| Restart method is supposed to begin a new environment session and return
       the initial observation, reward, and environment state (is the environment
       done?).

    2| Step method is going to take an action and return an observation, reward
       and environment state.

    3| Close method to clean up env.

    4| Actions have name and value.
       i:e
          ExecutionLink
            SchemaNode <action_name>
            NumberNode <value>
       NOTE: We are assuming the names of actions in the agent and the names of
             actions in the environment are in accordance. I am not sure if this
             is the right way to do it though.

    5| Reward is represented as
       i:e
          EvaluationLink
            PredicateNode "Reward"
            NumberNode <reward>

    6| Observations are stored in a python list.
       i:e
          [
            EvaluationLink
                PredicateNode <observation_name>
                <observation/s>
            , ...
          ]

    7| Environment state is `True` if environments' session has ended.

    """

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

    @abc.abstractmethod
    def close(self):
        pass
