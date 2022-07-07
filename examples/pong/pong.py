# Atari Pong Gym env
#
# WARNING: unmaintained

import gym
import random
import matplotlib.pyplot as plt
import numpy as np
from opencog.type_constructors import *
from opencog.spacetime import *

atomspace = AtomSpace()
set_default_atomspace(atomspace)

atomese_action_space = (
    SchemaNode("Noop"),
    SchemaNode("Fire"),
    SchemaNode("Go Right"),
    SchemaNode("Go Left"),
    SchemaNode("Fire Right"),
    SchemaNode("Fire Left"),
)

ATOMESE_TO_KEYS = {
    "Noop": 0,
    "Fire": 1,
    "Go Right": 2,
    "Go Left": 3,
    "Fire Right": 4,
    "Fire Left": 5,
}


def atomese_action_sample():
    return atomese_action_space[random.randint(0, 5)]


def to_atlocation(cord, obj):
    """
    Accepts center of Player, ball and opponent
    location convert and pass it to AtLocationLink
    Each object will be stored as such:

        (AtLocationLink
            (ConceptNode "Ball")
            (ListLink
                (ConceptNode "Pong"))
                (ListLink
                    (NumberNode 45)
                    (NumberNode 76)
                    (NumberNode 0)))

    returns a list of AtlocationLink for one frame
    """
    tmp = [NumberNode(str(n)) for n in cord]
    schema = AtLocationLink(
        ConceptNode(str(obj)),
        ListLink(ConceptNode("Pong"), ListLink(tmp[0], tmp[1], NumberNode("0"))),
    )

    return schema


def preprocess(obs):
    """
    Converts RGB observation into black and white
    white pixels represents the positon of ball and paddles

    """
    obs = obs[35:195][0:-1]
    obs[obs == 144] = 0
    obs[obs == 72] = 0
    obs[obs == 17] = 0
    return obs


def observation_to_atomese(observation):
    """
    Accepts observation from gym and converts
    it to atomese. Note that we convert only the
    white pixels.i.e the the x and y corrdinates
    of all white pixels in the game which represents
    the position of the paddle and the ball

    returns a List of EvaluationLinks
    """
    obs = preprocess(observation)
    player = []
    ball = []
    opp = []
    Player = np.array([92, 186, 92])
    Ball = np.array([236, 236, 236])
    Opponent = np.array([213, 130, 74])

    for i in range(159):
        for j in range(160):
            temp = obs[i][j]
            if (temp == Player).all():
                player.append([j, i])
            elif (temp == Ball).all():
                ball.append([j, i])

            elif (temp == Opponent).all():
                opp.append([j, i])
            else:
                pass

    beg = player[0]
    end = player.pop()
    midplayer = ((beg[0] + end[0]) / 2, (beg[1] + end[1]) / 2)
    schemaplayer = to_atlocation(midplayer, "Player")
    schemaball = SchemaNode("Null")
    schemaopp = SchemaNode("Null")

    if len(ball) != 0:
        beg = ball[0]
        end = ball.pop()
        midball = ((beg[0] + end[0]) / 2, (beg[1] + end[1]) / 2)
        schemaball = to_atlocation(midball, "Ball")

    if len(opp) != 0:
        beg = opp[0]
        end = opp.pop()
        midopp = ((beg[0] + end[0]) / 2, (beg[1] + end[1]) / 2)
        schemaopp = to_atlocation(midopp, "Opponent")
    return [schemaplayer, schemaball, schemaopp]


def reward_to_atomese(reward):
    rn = NumberNode(str(reward))
    return EvaluationLink(PredicateNode("Reward"), rn)


def action_to_gym(action):
    """
    Converts atomese type action into gym
    """
    if SchemaNode("Noop") == action:
        return 0
    if SchemaNode("Fire") == action:
        return 1
    if SchemaNode("Go Right") == action:
        return 2
    if SchemaNode("Go Left") == action:
        return 3
    if SchemaNode("Fire Right") == action:
        return 4
    if SchemaNode("Fire Left") == action:
        return 5


def dummy_atomese_agent(atomese_obser):
    """
    did not use observation at all
    Dummy agent
    """
    rand = random.randint(0, 5)
    return atomese_action_space[rand]


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    for i in range(100):
        env.reset()
        action = atomese_action_sample()
        for j in range(100):
            env.render()
            gym_action = action_to_gym(action)
            observation, reward, done, info = env.step(gym_action)
            atomese_ops = observation_to_atomese(observation)
            action = dummy_atomese_agent(atomese_ops)

    env.close()
