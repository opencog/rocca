import time

# OpenCog
from opencog.logger import log
from opencog.pln import *
from opencog.type_constructors import *
from opencog.utilities import set_default_atomspace

from rocca.agents import OpencogAgent
from rocca.agents.utils import agent_log
from rocca.envs.malmo_demo.flatworld_env import *
from rocca.envs.wrappers import MalmoWrapper
from opencog.ure import ure_logger
from rocca.envs.wrappers.utils import mk_action
from functools import wraps


class AbstractMalmoWrapper(MalmoWrapper):
    def __init__(self, missionXML, validate):
        MalmoWrapper.__init__(self, missionXML, validate)

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(ref, action, update_callback=None):
            # name, value = action.out[0], action.out[1]
            # ws = step(ref, name.name + " " + value.name, update_callback)
            name = action.out[0]
            ws = step(ref, name, update_callback)

            return MalmoWrapper.parse_world_state(ws)

        return wrapper

    @step_decorator.__func__
    def step(self, action, update_callback=None):
        try:
            func = globals()[action.name]
            func(self.agent_host)
            time.sleep(0.2)
            self.world_state = self.agent_host.getWorldState()

            for error in self.world_state.errors:
                print("Error: ", error.text)

            if update_callback is not None:
                update_callback(action, self.world_state)
        except RuntimeError as e:
            print("Error sending command:", e)
        return self.world_state

    

if __name__ == "__main__":
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)

    # Init loggers
    log.set_level("info")
    # log.set_sync(True)
    agent_log.set_level("debug")
    # agent_log.set_sync(True)
    ure_logger().set_level("debug")

    # 
    wrapped_env = AbstractMalmoWrapper(missionXML=missionXML, validate=True)

    # Create Goal
    pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
    ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

    # Create Action Space. The set of allowed actions an agent can take.
    # TODO take care of action parameters.
    action_space = {
        ExecutionLink(SchemaNode("go_to_the_key")), 
        ExecutionLink(SchemaNode("get_the_key")),
        ExecutionLink(SchemaNode("go_to_the_house")),
        ExecutionLink(SchemaNode("open_the_door")),
        ExecutionLink(SchemaNode("go_to_the_diamonds")),
        ExecutionLink(SchemaNode("collect_diamonds"))
    }

    cog_agent = OpencogAgent(wrapped_env, atomspace, action_space, pgoal, ngoal)


    # Training/learning loop
    lt_iterations = 3  # Number of learning-training iterations
    lt_period = 200  # Duration of a learning-training iteration
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
            "Action counter during {}th iteration:\n{}".format(i + 1, cog_agent.action_counter)
        )
