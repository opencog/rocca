import time

# OpenCog
from opencog.logger import log
from opencog.pln import *
from opencog.type_constructors import *
from opencog.utilities import set_default_atomspace
from opencog.ure import ure_logger


from rocca.agents import OpencogAgent
from rocca.agents.utils import TRUE_TV, agent_log
from rocca.envs.malmo_demo.collect_diamonds_env import *
from rocca.envs.wrappers import MalmoWrapper
from rocca.envs.wrappers.utils import mk_evaluation
from rocca.agents.utils import *
from functools import wraps
import os
import requests


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
            ExecutionLink(SchemaNode("go_to_key")),
            ExecutionLink(SchemaNode("go_to_house")),
            ExecutionLink(SchemaNode("go_to_diamonds")),
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
        self.cogscm_minimum_strength = 0.75
        self.cogscm_maximum_shannon_entropy = 1
        self.cogscm_maximum_differential_entropy = 0
        self.cogscm_maximum_variables = 0
        self.miner_maximum_iterations = 100000
        self.miner_maximum_variables = 9
        self.miner_minimum_support = 3
        self.conditional_conjunction_introduction = False
        self.expiry = 3
        self.visualize_cogscm = False
        self.mixture_model.complexity_penalty = 0.6
        self.mixture_model.delta = 1.0e-30
        self.mixture_model.compressiveness = 0.1
        self.mixture_model.weight_influence = 1.0

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
    ure_logger().set_level("debug")
    miner_log = MinerLogger(atomspace)
    miner_log.set_level("debug")

    #
    wrapped_env = AbstractMalmoWrapper(missionXML=missionXML, validate=True)

    cog_agent = FlatworldAgent(wrapped_env, atomspace)

    cogscm_atomspace_file = "cogscms_as.scm"
    saved_cogscm = os.path.isfile(cogscm_atomspace_file)
    if cog_agent.visualize_cogscm:
        # The address of the scheme REST endpoint
        PORT, IP_ADDRESS = "5000", "127.0.0.1"
        try:
            status_code = requests.get(
                "http://{}:{}/api/v1.1/atoms".format(IP_ADDRESS, PORT)
            )
        except:
            raise RuntimeError(
                "REST service is not started. Please run start_rest_service.py file first."
            )

    # Log all parameters of cag, useful for debugging
    cog_agent.log_parameters(level="debug")
    reward_t0 = cog_agent.record(cog_agent.initial_reward, 0, TRUE_TV)
    agent_log.info("Initial reward: {}".format(reward_t0))

    # Training/learning loop
    lt_iterations = 2  # Number of learning-training iterations
    lt_period = 50  # Duration of a learning-training iteration
    for i in range(lt_iterations):
        cog_agent.reset_action_counter()
        par = cog_agent.accumulated_reward  # Keep track of the reward before
        if i > 0:
            print(
                """
            **************************************
            * Applying PLN and Pattern Miner ... *
            **************************************

            """
            )
            if saved_cogscm:
                # load cogscm_atomspace
                cog_agent.load_cogscms_atomspace(cogscm_atomspace_file)
            else:
                # Discover patterns to make more informed decisions
                agent_log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
                cog_agent.learn()

            # Visualize discovered cogscms
            # set self.visualize_cogscm=True to enable visualization of the cogscms.
            # Note that, RESTAPI needs be running to be able to post to the scheme endpoint.
            # check start_rest_service.py file.
            if cog_agent.visualize_cogscm:
                post_to_restapi_scheme_endpoint(
                    None, cogscm_as=cog_agent.cogscms_atomspace
                )

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
        # Log all agent's atomspaces at the end (at fine level)
        agent_log_atomspace(cog_agent.atomspace, "fine", "cog_agent.atomspace")
        agent_log_atomspace(
            cog_agent.percepta_atomspace, "fine", "cog_agent.percepta_atomspace"
        )
        agent_log_atomspace(
            cog_agent.cogscms_atomspace, "fine", "cog_agent.cogscms_atomspace"
        )
        agent_log_atomspace(
            cog_agent.working_atomspace, "fine", "cog_agent.working_atomspace"
        )

    # Save cogscm_atomspace
    if not saved_cogscm:
        cog_agent.save_cogscms_atomspace("cogscms_as.scm")
