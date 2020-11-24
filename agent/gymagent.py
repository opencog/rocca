# Class with an agent to interact with OpenAI Gym environment

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
from opencog.atomspace import get_type, is_a
from opencog.type_constructors import *
from opencog.spacetime import *
from opencog.pln import *
from opencog.logger import Logger, log

# GymAgent
from .utils import *

#############
# Constants #
#############

TRUE_TV = TruthValue(1, 1)
DEFAULT_TV = TruthValue(1, 0)
X_ENABLED = 'DISPLAY' in os.environ

#########
# Class #
#########

class GymAgent:
    """Generic opencog gym agent to be derived.

    """

    def __init__(self, gym_env):
        self.atomspace = AtomSpace()
        set_default_atomspace(self.atomspace)
        self.env = gym_env
        self.observation = self.env.reset()
        self.step_count = 0
        self.accumulated_reward = 0
        self.percepta_record = ConceptNode("Percepta Record")

    def __del__(self):
        self.env.close()

    def record(self, atom, i, tv=None):
        """Timestamp and record an atom to the Percepta Record.

        That is add the following in the atomspace

        MemberLink (stv 1 1)
          AtTimeLink <tv>
            <atom>
            <i>
          <self.percepta_record>

        """

        return MemberLink(timestamp(atom, i, tv), self.percepta_record, tv=TRUE_TV)

    def gym_observation_to_atomese(self, observation):
        """Translate gym observation to Atomese, to be overloaded.

        The resulting translation does not need to have a TV or being
        timestamped, as this should be handled by the caller of that
        method.

        There is currently not standard for choosing the atomese
        representation but an example is given below.
        
        For instance if the observations are from the CartPole
        environment

        Observation               Min             Max
        -----------               ---             ---
        Cart Position             -4.8            4.8
        Cart Velocity             -Inf            Inf
        Pole Angle                -24 deg         24 deg
        Pole Velocity At Tip      -Inf            Inf

        One way to represent them would be

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

        A python list (not an atomese list) is returned with these 4
        Atomese observations.

        """

        return []


    def gym_reward_to_atomese(self, reward):
        """Translate gym reward to Atomese

        Evaluation
          Predicate "Reward"
          Number reward

        The reward representation is neither tv-set nor
        timestamped. It is up to the caller to do it.

        """

        rn = NumberNode(str(reward))
        return EvaluationLink(PredicateNode("Reward"), rn)


    def atomese_action_space(self):
        """Return the set of possible atomese actions. To be overloaded.

        Atomese actions are typically represented as SchemaNode. For
        now action parameters are ignored.

        """

        return {}


    def atomese_action_to_gym(self, action):
        """Map atomese actions to gym actions. To be overloaded.

        For instance in CartPole-v1 the mapping is as follows

        SchemaNode("Go Left") -> 0
        SchemaNode("Go Right") -> 1

        """

        return 0


    def make_goal(self):

        """Define the goal of the current iteration.

        By default the goal of the current iteration is to have a
        reward of 1.

        Evaluation
          Predicate "Reward"
          Number 1

        """

        return self.gym_reward_to_atomese(1)


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

        For now it is assumed that <action> is fully grounded.

        """

        # TODO
        return []


    def deduce(self, cogscms):
        """Return an action distribution given a list cognitive schematics.

        The action distribution is actually a second order
        distribution, i.e. each action is weighted with a truth value.
        Such truth value, called the action truth value, corresponds
        to the second order probability of acheiving the goal if the
        action is taken right now.

        Formally the meaning of the action truth value can be
        expressed as follows:

        Subset <action-tv>
          SubjectivePaths
            AtTime
              Execution
                <action>
              <i>
          SubjectivePaths
            AtTime
              <goal>
              <i + offset>

        where

        SubjectivePaths
          <P>

        is the set of all subjective paths compatible with <P>, and a
        subjective path is a sequence of sujective states (atomspace
        snapshots) indexed by time.

        In order to infer such action truth value one needs to perform
        deduction (or modus ponens) on the cognitive schematics,
        combining the probability of the context being currently true.

        """

        log.debug("deduce(cogscms={})".format(cogscms))

        # For each cognitive schematic estimate the probability of its
        # context to be true and multiply it by the truth value of the
        # cognitive schematic, then calculate its weight based on
        # https://github.com/ngeiswei/papers/blob/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf
        # and use it to perform a Bayesian Model Averaging to obtain
        # the second order distribution of each action.
        #
        # Important Notes:

        # 1. Adding an unknown component (with flat or such prior) in
        #    the BMA can flatten the resulting distribution and be
        #    used to user-tune exploration vs exploitation in a
        #    justified manner. This is probably equivalent to (or
        #    better than) Epsilon-best. Ultimately the weight of such
        #    unknown component should be calculated as the unknown
        #    rest in the Solomonoff mixture.
        #
        # 2. We actually don't need to build the mixture model, rather
        #    we just need to hand the convex combination of models to
        #    the decide function which will do the Thompson sampling.
        #
        # 3. It's unclear if the probability of the context should
        #    altern the model TV or its weight. We need to think of
        #    generalizing the inactive models as well.

        # For now discretize context truth into valid and invalid and
        # only consider valid cognitive schematics in the BMA
        # (Bayesian Model Averaging). It's not entirely clear what to
        # do with the invalid cognitive schematics, maybe they should
        # be taken into account to lower the confidence of the final
        # result, as they allegedly exert an unknown influence (via
        # their invalid parts).
        ctx_tv = lambda cogscm : \
            get_context_actual_truth(self.atomspace, cogscm, self.step_count)
        valid_cogscms = [cogscm for cogscm in cogscms if 0.9 < ctx_tv(cogscm).mean]
        log.debug("valid_cogscms = {}".format(valid_cogscms))

        # For now we have a uniform weighting across valid cognitive
        # schematics.
        actdist = omdict([(get_action(cogscm), (1.0, cogscm.tv))
                          for cogscm in valid_cogscms])
        log.debug("actdist-1 = {}".format(actdist))

        # Add an unknown component for each action. For now its weight
        # is constant, delta, but ultimately is should be calculated
        # as a rest in the Solomonoff mixture.
        delta = 0.5
        for action in self.atomese_action_space():
            actdist.add(action, (delta, DEFAULT_TV))
        log.debug("actdist-2 = {}".format(actdist))

        return actdist


    def decide(self, actdist):
        """Select the next action to enact from an action distribution.

        The action is selected from the action distribution, a list of
        pairs (action, tv), obtained from deduce.  The selection uses
        Thompson sampling leveraging the second order distribution to
        balance exploitation and exploration. See
        http://auai.org/uai2016/proceedings/papers/20.pdf for more
        details about Thompson Sampling.

        """

        # Select the pair of action and its first order probability of
        # success according to Thompson sampling
        (action, pblty) = thompson_sample(actdist)

        # Return the action (we don't need the probability for now)
        return (action, pblty)


    def step(self):
        """Run one step of observation, decision and env update
        """
        
        # Translate to atomese and timestamp observations
        atomese_obs = self.gym_observation_to_atomese(self.observation)
        log.debug("atomese_obs = {}".format(atomese_obs))
        obs_record = [self.record(o, self.step_count, tv=TRUE_TV)
                      for o in atomese_obs]
        log.debug("obs_record = {}".format(obs_record))

        # Make the goal for that iteration
        goal = self.make_goal()
        log.debug("goal = {}".format(goal))

        # Render the environment if X is running
        if X_ENABLED:
            self.env.render()

        # Plan, i.e. come up with cognitive schematics as plans.  Here the
        # goal expiry is 1, i.e. set for the next iteration.
        expiry = 1
        css = self.plan(goal, expiry)
        log.debug("css = {}".format(css))

        # Deduce the action distribution
        actdist = self.deduce(css)
        log.debug("actdist = {}".format(actdist))

        # Select the next action
        action, pblty = self.decide(actdist)
        log.debug("(action={}, pblty={})".format(action, pblty))

        # Timestamp the action that is about to be executed
        action_exec = ExecutionLink(action)
        action_exec_record = self.record(action_exec, self.step_count, tv=TRUE_TV)
        log.debug("action_exec_record = {}".format(action_exec_record))

        # Convert atomese action to openai gym action
        gym_action = self.atomese_action_to_gym(action)
        log.debug("gym_action = {}".format(gym_action))

        # Increase the step count and run the next step of the environment
        self.step_count += 1
        self.observation, reward, done, info = self.env.step(gym_action)
        self.accumulated_reward += reward
        log.debug("observation = {}".format(self.observation))
        log.debug("reward = {}".format(reward))
        log.debug("accumulated reward = {}".format(self.accumulated_reward))
        log.debug("info = {}".format(info))

        # Translate reward to atomese and timestamp it
        atomese_reward = self.gym_reward_to_atomese(reward)
        reward_record = self.record(atomese_reward, self.step_count, tv=TRUE_TV)
        log.debug("reward_record = {}".format(reward_record))

        if done:
            return False

        return True
