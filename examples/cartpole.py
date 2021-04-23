# Define an agent for the CartPole env with GymAgent

##############
# Initialize #
##############

# Python
import os
import time
from orderedmultidict import omdict

# OpenCog
from opencog.atomspace import AtomSpace, TruthValue
from opencog.atomspace import types
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.ure import ure_logger

# OpenAI Gym
import gym
env = gym.make('CartPole-v1')
# Uncomment the following to get a description of env
# help(env.unwrapped)

# OpenCog Gym
from rocca.agents import OpencogAgent
from rocca.agents.utils import *
from rocca.envs.wrappers import GymWrapper

####################
# CartPole Wrapper #
####################

class CartPoleWrapper(GymWrapper):
    def __init__(self, env):
        action_list = ["Go Left", "Go Right"]
        GymWrapper.__init__(self, env, action_list)

    def labeled_observations(self, space, obs, sbs=""):
        """Translate gym observation to Atomese

        There are 4 observations (taken from CartPoleEnv help)

        Observation               Min             Max
        -----------               ---             ---
        Cart Position             -4.8            4.8
        Cart Velocity             -Inf            Inf
        Pole Angle                -24 deg         24 deg
        Pole Velocity At Tip      -Inf            Inf

        They are represented in atomese as follows

        Evaluation
          Predicate "Cart Position"
          Number CP

        Evaluation
          Predicate "Cart Velocity"
          Number CV

        Evaluation
          Predicate "Pole Angle"
          Number PA

        Evaluation
          Predicate "Pole Velocity At Tip"
          Number PVAT

        Note that the observations are neither tv-set nor
        timestamped. It is up to the caller to do it.

        A python list (not an atomese list) is returned with these 4
        Atomese observations.

        """

        cp = NumberNode(str(obs[0]))
        cv = NumberNode(str(obs[1]))
        pa = NumberNode(str(obs[2]))
        pvat = NumberNode(str(obs[3]))

        return [EvaluationLink(PredicateNode("Cart Position"), cp),
                EvaluationLink(PredicateNode("Cart Velocity"), cv),
                EvaluationLink(PredicateNode("Pole Angle"), pa),
                EvaluationLink(PredicateNode("Cart Velocity At Tip"), pvat)]


##################
# CartPole Agent #
##################

class CartPoleAgent(OpencogAgent):
    def __init__(self, env):
        # Create Action Space. The set of allowed actions an agent can take.
        # TODO take care of action parameters.
        action_space = {ExecutionLink(SchemaNode(a)) for a in env.action_list}

        # Create Goal
        pgoal = EvaluationLink(PredicateNode("Reward"), NumberNode("1"))
        ngoal = EvaluationLink(PredicateNode("Reward"), NumberNode("0"))

        # Call super ctor
        OpencogAgent.__init__(self, env, action_space, pgoal, ngoal)

    def plan(self, goal, expiry):
        """Plan the next actions given a goal and its expiry time offset

        Return a python list of cognivite schematics.  Whole cognitive
        schematics are output (instead of action plans) in order to
        make a decision based on their truth values.  Alternatively it
        could return a pair (action plan, tv), where tv has been
        evaluated to take into account the truth value of the context
        as well (which would differ from the truth value of rule in
        case the context is uncertain).

        The format for a cognitive schematic is as follows

        PredictiveImplicationScope <tv>
          <vardecl>
          <expiry>
          And (or SimultaneousAnd?)
            <context>
            Execution
              <action>
              <input> [optional]
              <output> [optional]
          <goal>

        """

        # For now we provide 2 hardwired rules
        #
        # 1. Push cart to the left (0) if angle is negative
        # 2. Push cart to the right (1) if angle is positive
        #
        # with some arbitrary truth value (stv 0.9, 0.1)
        angle = VariableNode("$angle")
        numt = TypeNode("NumberNode")
        time_offset = to_nat(1)
        pole_angle = PredicateNode("Pole Angle")
        go_right = SchemaNode("Go Right")
        go_left = SchemaNode("Go Left")
        reward = PredicateNode("Reward")
        epsilon = NumberNode("0.01")
        mepsilon = NumberNode("-0.01")
        unit = NumberNode("1")
        hTV = TruthValue(0.9, 0.1)  # High TV
        lTV = TruthValue(0.1, 0.1)  # Low TV

        # PredictiveImplicationScope <high TV>
        #   TypedVariable
        #     Variable "$angle"
        #     Type "NumberNode"
        #   Time "1"
        #   And
        #     Evaluation
        #       Predicate "Pole Angle"
        #       Variable "$angle"
        #     GreaterThan
        #       Variable "$angle"
        #       0
        #     Execution
        #       Schema "Go Right"
        #   Evaluation
        #     Predicate "Reward"
        #     Number "1"
        cs_rr = \
            PredictiveImplicationScopeLink(
                TypedVariableLink(angle, numt),
                time_offset,
                AndLink(
                    # Context
                    EvaluationLink(pole_angle, angle),
                    GreaterThanLink(angle, epsilon),
                    # Action
                    ExecutionLink(go_right)),
                # Goal
                EvaluationLink(reward, unit),
                # TV
                tv=hTV)

        # PredictiveImplicationScope <high TV>
        #   TypedVariable
        #     Variable "$angle"
        #     Type "NumberNode"
        #   Time "1"
        #   And
        #     Evaluation
        #       Predicate "Pole Angle"
        #       Variable "$angle"
        #     GreaterThan
        #       0
        #       Variable "$angle"
        #     Execution
        #       Schema "Go Left"
        #   Evaluation
        #     Predicate "Reward"
        #     Number "1"
        cs_ll = \
            PredictiveImplicationScopeLink(
                TypedVariableLink(angle, numt),
                time_offset,
                AndLink(
                    # Context
                    EvaluationLink(pole_angle, angle),
                    GreaterThanLink(mepsilon, angle),
                    # Action
                    ExecutionLink(go_left)),
                # Goal
                EvaluationLink(reward, unit),
                # TV
                tv=hTV)

        # To cover all possibilities we shouldn't forget the complementary
        # actions, i.e. going left when the pole is falling to the right
        # and such, which should make the situation worse.

        # PredictiveImplicationScope <low TV>
        #   TypedVariable
        #     Variable "$angle"
        #     Type "NumberNode"
        #   Time "1"
        #   And (or SimultaneousAnd?)
        #     Evaluation
        #       Predicate "Pole Angle"
        #       Variable "$angle"
        #     GreaterThan
        #       Variable "$angle"
        #       0
        #     Execution
        #       Schema "Go Left"
        #   Evaluation
        #     Predicate "Reward"
        #     Number "1"
        cs_rl = \
            PredictiveImplicationScopeLink(
                TypedVariableLink(angle, numt),
                time_offset,
                AndLink(
                    # Context
                    EvaluationLink(pole_angle, angle),
                    GreaterThanLink(angle, epsilon),
                    # Action
                    ExecutionLink(go_left)),
                # Goal
                EvaluationLink(reward, unit),
                # TV
                tv=lTV)

        # PredictiveImplicationScope <low TV>
        #   TypedVariable
        #     Variable "$angle"
        #     Type "NumberNode"
        #   Time "1"
        #   And (or SimultaneousAnd?)
        #     Evaluation
        #       Predicate "Pole Angle"
        #       Variable "$angle"
        #     GreaterThan
        #       0
        #       Variable "$angle"
        #     Execution
        #       Schema "Go Right"
        #   Evaluation
        #     Predicate "Reward"
        #     Number "1"
        cs_lr = \
            PredictiveImplicationScopeLink(
                TypedVariableLink(angle, numt),
                time_offset,
                AndLink(
                    # Context
                    EvaluationLink(pole_angle, angle),
                    GreaterThanLink(mepsilon, angle),
                    # Action
                    ExecutionLink(go_right)),
                # Goal
                EvaluationLink(reward, unit),
                # TV
                tv=lTV)

        # Ideally we want to return only relevant cognitive schematics
        # (i.e. with contexts probabilistically currently true) for
        # now however we return everything and let to the deduction
        # process deal with it, as it should be able to.
        return [cs_ll, cs_rr, cs_rl, cs_lr]


########
# Main #
########
def main():
    # Init loggers
    log = create_logger("opencog.log")
    log.set_level("debug")
    log.set_sync(False)
    agent_log.set_level("fine")
    agent_log.set_sync(False)
    ure_logger().set_level("debug")
    ure_logger().set_sync(False)

    # Set main atomspace
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)

    # Wrap environment
    wrapped_env = CartPoleWrapper(env)

    # Instantiate CartPoleAgent, and tune parameters
    cpa = CartPoleAgent(wrapped_env)
    cpa.delta = 1.0e-16

    # Run control loop
    while (cpa.step() or True):
        time.sleep(0.1)
        log.info("step_count = {}".format(cpa.step_count))


if __name__ == "__main__":
    main()
