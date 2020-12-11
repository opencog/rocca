# Define an agent for the Chase env with OpencogAgent

##############
# Initialize #
##############

# Python
import time

# OpenAI Gym
import gym
# OpenCog
from opencog.pln import *

# OpenCog Gym
from agent.OpencogAgent import OpencogAgent
from agent.utils import *
from envs.wrappers import GymWrapper

env = gym.make('Chase-v0')
# Uncomment the following to get a description of env
# help(env.unwrapped)

###############
# Chase Agent #
###############

class ChaseAgent(OpencogAgent):
    def __init__(self, env, action_space, p_goal, n_goal):
        OpencogAgent.__init__(self, env, action_space, p_goal, n_goal)

    # def learn(self):
    #     agent_log.fine("ChaseAgent.learn()")

    #     # For now hardwire various plans of different timescales
    #     agent_position = PredicateNode("0-Discrete")
    #     pellet_position = PredicateNode("1-Discrete")
    #     position = VariableNode("$position")
    #     left_pos = NumberNode("0")
    #     right_pos = NumberNode("1")
    #     concept_t = TypeNode("ConceptNode")
    #     number_t = TypeNode("NumberNode")
    #     eat = SchemaNode("Eat")
    #     go_right = SchemaNode("Go Right")
    #     go_left = SchemaNode("Go Left")
    #     reward = PredicateNode("Reward")
    #     unit = NumberNode("1")
    #     vhTV = TruthValue(1.0, 0.1)  # Very high TV

    #     # If agent position equals pellet position then eating brings
    #     # a reward
    #     eat_cogscm = \
    #         PredictiveImplicationScopeLink(
    #             TypedVariableLink(position, number_t),
    #             to_nat(1),
    #             AndLink(
    #                 # Context
    #                 EvaluationLink(agent_position, position),
    #                 EvaluationLink(pellet_position, position),
    #                 # Action
    #                 ExecutionLink(eat)),
    #             # Goal
    #             EvaluationLink(reward, unit),
    #             # TV
    #             tv=vhTV)
    #     agent_log.fine("eat_cogscm = {}".format(eat_cogscm))

    #     # If agent position is Left and pellet position is Right then
    #     # going right and eating brings a reward
    #     go_right_eat_cogscm = \
    #         PredictiveImplicationScopeLink(
    #             VariableSet(),
    #             to_nat(1),
    #             AltSequentialAndLink(
    #                 to_nat(1),
    #                 AndLink(
    #                     # Context
    #                     EvaluationLink(agent_position, left_pos),
    #                     EvaluationLink(pellet_position, right_pos),
    #                     # Action 1
    #                     ExecutionLink(go_right)),
    #                 # Action 2
    #                 ExecutionLink(eat)),
    #             # Goal
    #             EvaluationLink(reward, unit),
    #             # TV
    #             tv=vhTV)
    #     agent_log.fine("go_right_eat_cogscm = {}".format(go_right_eat_cogscm))

    #     # If agent position is Right and pellet position is Left then
    #     # going left and eating brings a reward
    #     go_left_eat_cogscm = \
    #         PredictiveImplicationScopeLink(
    #             VariableSet(),
    #             to_nat(1),
    #             AltSequentialAndLink(
    #                 to_nat(1),
    #                 AndLink(
    #                     # Context
    #                     EvaluationLink(agent_position, right_pos),
    #                     EvaluationLink(pellet_position, left_pos),
    #                     # Action 1
    #                     ExecutionLink(go_left)),
    #                 # Action 2
    #                 ExecutionLink(eat)),
    #             # Goal
    #             EvaluationLink(reward, unit),
    #             # TV
    #             tv=vhTV)
    #     agent_log.fine("go_left_eat_cogscm = {}".format(go_left_eat_cogscm))

    #     self.cognitive_schematics = set([eat_cogscm,
    #                                      go_right_eat_cogscm,
    #                                      go_left_eat_cogscm])

if __name__ == "__main__":
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)
    # Wrap environment
    # Allowed_actions is not required if the gym environment's action
    # space is labeled a.k.a space.Dict.
    allowed_actions = ["Go Left", "Go Right", "Stay", "Eat"]
    wrapped_env = GymWrapper(env, allowed_actions)

    # Create Goal
    pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
    ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

    # Create Action Space. The set of allowed actions an agent can take.
    # TODO take care of action parameters.
    action_space = {ExecutionLink(SchemaNode("Go Left")),
                    ExecutionLink(SchemaNode("Go Right")),
                    ExecutionLink(SchemaNode("Stay")),
                    ExecutionLink(SchemaNode("Eat"))}

    # ChaseAgent
    ca = ChaseAgent(wrapped_env, action_space, pgoal, ngoal)

    # Training/learning loop
    lt_iterations = 2           # Number of learning-training iterations
    lt_period = 200             # Duration of a learning-training iteration
    for i in range(lt_iterations):
        ca.reset_action_counter()
        par = ca.accumulated_reward # Keep track of the reward before
        # Discover patterns to make more informed decisions
        agent_log.info("Start learning ({}/{})".format(i + 1, lt_iterations))
        ca.learn()
        # Run agent to accumulate percepta
        agent_log.info("Start training ({}/{})".format(i + 1, lt_iterations))
        for j in range(lt_period):
            ca.step()
            time.sleep(0.01)
            log.info("step_count = {}".format(ca.step_count))
        nar = ca.accumulated_reward - par
        agent_log.info("Accumulated reward during {}th iteration = {}".format(i + 1, nar))
        agent_log.info("Action counter during {}th iteration:\n{}".format(i+1, ca.action_counter))
