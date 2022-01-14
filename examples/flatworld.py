import time

# OpenCog
from opencog.logger import log
from opencog.pln import *
from opencog.type_constructors import *
from opencog.utilities import set_default_atomspace

from rocca.agents import OpencogAgent
from rocca.agents.utils import TRUE_TV, agent_log
from rocca.envs.malmo_demo.flatworld_env import *
from rocca.envs.wrappers import MalmoWrapper
from opencog.ure import ure_logger
from rocca.envs.wrappers.utils import mk_evaluation
from functools import wraps


class AbstractMalmoWrapper(MalmoWrapper):
    def __init__(self, missionXML, validate):
        MalmoWrapper.__init__(self, missionXML, validate)

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(ref, action, update_callback=None):
            name = action.out[0]
            return step(ref, name, update_callback)

        return wrapper

    @step_decorator.__func__
    def step(self, abstract_action, update_callback=None):
        try:
            func = globals()[abstract_action.name]
            observations, reward, done = func(self.agent_host)
            time.sleep(0.2)
            obs_list = []
            if observations:
                for k in observations:
                    if isinstance(observations[k], list):
                        obs_list.append(mk_evaluation(k, *observations[k]))
                    else:
                        obs_list.append(mk_evaluation(k, observations[k]))
        except RuntimeError as e:
            print("Error sending command:", e)
        return obs_list, mk_evaluation("Reward", reward), done


class FlatworldAgent(OpencogAgent):
    def __init__(self, env, atomspace):
        set_default_atomspace(atomspace)

        # Create Action Space. The set of allowed actions an agent can take.
        # TODO take care of action parameters.
        action_space = {
            ExecutionLink(SchemaNode("go_to_the_key")),
            ExecutionLink(SchemaNode("go_to_the_house")),
            ExecutionLink(SchemaNode("go_to_the_diamonds")),
        }
        # Create Goal
        pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
        ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

        # Call super ctor
        OpencogAgent.__init__(self, env, atomspace, action_space, pgoal, ngoal)

        # Overwrite some OpencogAgent parameters
        self.polyaction_mining = False
        self.monoaction_general_succeedent_mining = True
        self.temporal_deduction = True
        self.cogscm_min_strength = 0.99
        self.cogscm_max_variables = 0
        # Todo: restart the environment to get an initial reward
        self.initial_reward = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))


if __name__ == "__main__":
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)

    # Init loggers
    log.set_level("info")
    # log.set_sync(True)
    agent_log.set_level("fine")
    # agent_log.set_sync(True)
    ure_logger().set_level("info")

    #
    wrapped_env = AbstractMalmoWrapper(missionXML=missionXML, validate=True)

    cog_agent = FlatworldAgent(wrapped_env, atomspace)
    reward_t0 = cog_agent.record(cog_agent.initial_reward, 0, TRUE_TV)
    agent_log.info("Initial reward: {}".format(reward_t0))

    # Training/learning loop
    lt_iterations = 2  # Number of learning-training iterations
    lt_period = 50  # Duration of a learning-training iteration
    for i in range(lt_iterations):
        cog_agent.reset_action_counter()
        par = cog_agent.accumulated_reward  # Keep track of the reward before
        # Discover patterns to make more informed decisions
        agent_log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        cog_agent.learn()
        # Run agent to accumulate percepta
        agent_log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            cog_agent.control_cycle()
            time.sleep(0.01)
            log.info("step_count = {}".format(cog_agent.cycle_count))
        nar = cog_agent.accumulated_reward - par
        agent_log.info(
            "Accumulated reward during {}th iteration = {}".format(i + 1, nar)
        )
        agent_log.info(
            "Action counter during {}th iteration:\n{}".format(
                i + 1, cog_agent.action_counter
            )
        )
