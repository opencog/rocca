# OpenCog wrapper around OpenAI Gym.
#
# Highly experimental at this stage.  For now we write an ad-hoc
# wrapper for the CartPole-v0 env, latter on this wrapper will be
# generalized to any env, and the user will have to overload of the
# various methods of that wrapper for specific env.

##############
# Initialize #
##############

# OpenCog
from opencog.atomspace import AtomSpace, TruthValue
from opencog.atomspace import types
from opencog.type_constructors import *
from opencog.nlp_types import *
a = AtomSpace()
set_default_atomspace(a)

# OpenAI Gym
import gym
env = gym.make('CartPole-v1')
# Uncomment the following to get a description of env
# help(env.unwrapped)

# Other
import os

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


def make_goal(ci):
    """Define the goal of the current iteration.

    Informally the goal of the current iteration is to have a reward
    of 1 for the next iteration.

    TODO: the goal should probably not be timestamped at this point.
    Instead a "deadline" should be provided to the planner.

    AtTime
      Evaluation
        Predicate "Reward"
        Number 1
      TimeNode str(ci + 1)

    """

    return timestamp(reward_to_atomese(1), ci + 1)


def timestamp(atom, i, tv=None):
    """Timestamp a given atom.  Optionally set its TV

    AtTimeLink tv               # if tv is provided
      atom
      TimeNode str(i)

    """

    return AtTimeLink(atom, TimeNode(str(i)), tv=tv)


def plan(goal):
    """Plan the next actions given a goal.

    Return a python list of cognivite schematics.  Whole cognitive
    schematics are output instead of action plans in order to make a
    decision based on their truth values.  Alternatively it could
    return a pair (action plan, tv), where tv has been evaluated to
    take into account the truth value of the context as well (which
    would differ from the truth value of rule in case the context is
    uncertain).

    PredictiveImplication

    If the goal has a timestamp (like the next iteration), such as

    AtTime
      atemporal-goal
      T + 1

    given that the current time is T, then a cognitive schematics is

    NEXT

    Subset
      SubjectivePaths
        CrisptLogicalClosure
          AtTime
            And
              context
              action-plan
            T
      SubjectivePaths
        CrisptLogicalClosure
          AtTime
            atemporal-goal
            T + 1

    """

    # NEXT
    return None


def decide(cogschs):
    """Select the next action given a list of cognitive schematics.

    The action selection uses Thomspon sampling leveraging the second
    order distribution of the cognitive schematics, combined with the
    context if not completely certain, to balance exploitation and
    exploration.

    """

    # NEXT
    return env.action_space.sample()


###########
# OpenPsi #
###########
import time
from opencog.openpsi import OpenPsi

op = OpenPsi(a)

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

    # Plan, i.e. come up with cognitive schematics as plans
    cogschs = plan(goal)
    print("cogschs =", cogschs)

    # Select the next action
    action = decide(cogschs)
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
