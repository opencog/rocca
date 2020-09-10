# OpenCog wrapper around OpenAI Gym.
#
# Highly experimental at this stage.  For now we write an ad-hoc
# wrapper for the CartPole-v0 env, latter on this wrapper will be
# generalized to any env, and the user will have to overload of the
# various methods of that wrapper for specific env.

##############
# Initialize #
##############

# Python
import os
import time
import random

# OpenCog
from opencog.atomspace import AtomSpace, TruthValue
from opencog.atomspace import types
from opencog.type_constructors import *
from opencog.pln import *
from opencog.scheme_wrapper import *
a = AtomSpace()
set_default_atomspace(a)

# OpenPsi
from opencog.openpsi import OpenPsi
op = OpenPsi(a)

# OpenAI Gym
import gym
env = gym.make('CartPole-v1')
# Uncomment the following to get a description of env
# help(env.unwrapped)

#############
# Constants #
#############

TRUE_TV = TruthValue(1, 1)
X_ENABLED = 'DISPLAY' in os.environ

#############
# Functions #
#############

def observation_to_atomese(observation):
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

    Note that the observations are neither tv-set nor timestamped. It
    is up to the caller to do it.

    A python list (not an atomese list) is returned with these 4
    Atomese observations.

    """

    cp = NumberNode(str(observation[0]))
    cv = NumberNode(str(observation[1]))
    pa = NumberNode(str(observation[2]))
    pvat = NumberNode(str(observation[3]))

    return [EvaluationLink(PredicateNode("Cart Position"), cp),
            EvaluationLink(PredicateNode("Cart Velocity"), cv),
            EvaluationLink(PredicateNode("Pole Angle"), pa),
            EvaluationLink(PredicateNode("Cart Velocity At Tip"), pvat)]


def reward_to_atomese(reward):
    """Translate gym reward to Atomese

    Evaluation
      Predicate "Reward"
      Number reward

    The reward representation is neither tv-set nor timestamped. It is
    up to the caller to do it.

    """

    rn = NumberNode(str(reward))
    return EvaluationLink(PredicateNode("Reward"), rn)


def action_to_gym(action):
    """Map atomese actions to gym actions

    In CartPole-v1 the mapping is as follows

    SchemaNode("Go Right") -> 0
    SchemaNode("Go Left") -> 1

    """

    if SchemaNode("Go Right") == action:
        return 0
    if SchemaNode("Go Left") == action:
        return 1


def get_action(cs):
    """extract the action of a cognitive schematic.

    Given a cognitive schematic of that format

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

    extract <action>. Ideally the input and output should also be
    extracted, for now only the action.

    """

    cnjs = cs.get_out()[2].get_out()
    execution = next(x for x in cnjs if x.type == types.ExecutionLink)
    return execution.get_out()[0]


def make_goal(ci):
    """Define the goal of the current iteration.

    Here the goal of the current iteration is to have a reward of 1.

    Evaluation
      Predicate "Reward"
      Number 1

    """

    return reward_to_atomese(1)


def timestamp(atom, i, tv=None):
    """Timestamp a given atom.  Optionally set its TV

    AtTimeLink tv               # if tv is provided
      atom
      TimeNode str(i)

    """

    return AtTimeLink(atom, TimeNode(str(i)), tv=tv)


def plan(goal, expiry):
    """Plan the next actions given a goal and its expiry time offset

    Return a python list of cognivite schematics.  Whole cognitive
    schematics are output (instead of action plans) in order to make a
    decision based on their truth values.  Alternatively it could
    return a pair (action plan, tv), where tv has been evaluated to
    take into account the truth value of the context as well (which
    would differ from the truth value of rule in case the context is
    uncertain).

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
    time_offset = NumberNode(str(expiry))
    pole_angle = PredicateNode("Pole Angle")
    go_right = SchemaNode("Go Right")
    go_left = SchemaNode("Go Left")
    reward = PredicateNode("Reward")
    zero = NumberNode("0")
    unit = NumberNode("1")
    TV = TruthValue(0.9, 0.1)

    # PredictiveImplicationScope
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
    #       Schema "Go Right"
    #   Evaluation
    #     Predicate "Reward"
    #     Number "1"
    cs_r = \
        PredictiveImplicationScopeLink(
            TypedVariableLink(angle, numt),
            time_offset,
            AndLink(
                # Context
                EvaluationLink(pole_angle, angle),
                GreaterThanLink(angle, zero),
                # Action
                ExecutionLink(go_right)),
            # Goal
            EvaluationLink(reward, unit),
            # TV
            tv=TV)

    # PredictiveImplicationScope
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
    #       Schema "Go Left"
    #   Evaluation
    #     Predicate "Reward"
    #     Number "1"
    cs_l = \
        PredictiveImplicationScopeLink(
            TypedVariableLink(angle, numt),
            time_offset,
            AndLink(
                # Context
                EvaluationLink(pole_angle, angle),
                GreaterThanLink(zero, angle),
                # Action
                ExecutionLink(go_left)),
            # Goal
            EvaluationLink(reward, unit),
            # TV
            tv=TV)

    # Ideally we want to return only relevant cognitive schematics
    # (i.e. with contexts probabilistically currently true) for now
    # however we return everything and let to the decision progress
    # deal with it, as it should be able to.
    return [cs_l, cs_r]


def decide(css):
    """Select the next action given a list of cognitive schematics.

    The action selection uses Thomspon sampling leveraging the second
    order distribution of the cognitive schematics, combined with the
    context if not completely certain, to balance exploitation and
    exploration.

    """

    # For now randomly sample from the list of cognitive schematics
    cs = random.choice(css)

    # Extract action from the selected cognitive schematics
    action = get_action(cs)

    # Translate atomese action to gym action
    return action_to_gym(action)


observation = env.reset()
cartpole_step_count =0
def cartpole_step():
    global observation
    global cartpole_step_count
    i = cartpole_step_count
    time.sleep(1)

    # Translate to atomese and timestamp observations
    atomese_obs = observation_to_atomese(observation)
    timestamped_obs = [timestamp(o, i, tv=TRUE_TV) for o in atomese_obs]
    print("timestamped_obs =", timestamped_obs)

    # Make the goal for that iteration
    goal = make_goal(i)
    print("goal =", goal)

    # Render the environment if X is running
    if X_ENABLED:
        env.render()

    # Plan, i.e. come up with cognitive schematics as plans.  Here the
    # goal expiry is 1, i.e. set for the next iteration.
    expiry = 1
    css = plan(goal, expiry)
    print("css =", css)

    # Select the next action
    action = decide(css)
    print("action =", action)

    # Run the next step of the environment
    observation, reward, done, info = env.step(action)
    print("observation =", observation)
    print("reward =", reward)
    print("info =", info)

    # Translate to atomese and timestamp reward
    timestamped_reward = timestamp(reward_to_atomese(reward), i, tv=TRUE_TV)
    print("timestamped_reward =", timestamped_reward)

    cartpole_step_count += 1
    if done:
        print("Stopping the openpsi loop")
        env.close()
        return TruthValue(0, 1)

    return TruthValue(1, 1)


cartpole_stepper = EvaluationLink(
    GroundedPredicateNode("py: cartpole_step"), ListLink()
)


cartpole_component = op.create_component("cartpole", cartpole_stepper)

########
# Main #
########
def main():
    op.run(cartpole_component)

# Launching main() from here produces a Segmentation fault
# if __name__ == "__main__":
#     main()
