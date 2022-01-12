import os
import time

from rocca.envs.wrappers.malmo_wrapper import MalmoWrapper
from rocca.envs.wrappers.utils import *
from builtins import range
import json
import math
import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


def drawBlock(x, y, z, type):
    return """<DrawBlock x='{0}' y='{1}' z='{2}' type='{3}'/>""".format(x, y, z, type)


def GenCuboid(x1, y1, z1, x2, y2, z2, blocktype):
    return (
        '<DrawCuboid x1="'
        + str(x1)
        + '" y1="'
        + str(y1)
        + '" z1="'
        + str(z1)
        + '" x2="'
        + str(x2)
        + '" y2="'
        + str(y2)
        + '" z2="'
        + str(z2)
        + '" type="'
        + blocktype
        + '"/>'
    )


def drawLine(x1, y1, z1, x2, y2, z2, blocktype, face):
    return (
        '<DrawLine x1="'
        + str(x1)
        + '" y1="'
        + str(y1)
        + '" z1="'
        + str(z1)
        + '" x2="'
        + str(x2)
        + '" y2="'
        + str(y2)
        + '" z2="'
        + str(z2)
        + '" type="'
        + blocktype
        + '" face="'
        + face
        + '"/>'
    )


def build_house(x, y, z, width, length, height, blocktype):
    x_front = x
    x_back = x + length if x >= 0 else x - length
    z_left = z + width if z >= 0 else z - width
    z_right = z

    genstring = ""
    y_cur = y
    for i in range(height):
        z_cur = z
        x_cur = x
        for j in range(width):
            genstring = genstring + drawBlock(x_front, y_cur, z_cur, "glass") + "\n"
            genstring = (
                genstring + drawBlock(x_back, y_cur, z_cur, "diamond_block") + "\n"
            )
            genstring = (
                genstring + drawBlock(x_back - 1, y_cur, z_cur, blocktype) + "\n"
            )
            z_cur = z_cur + 1 if z_cur >= 0 else z_cur - 1

        for k in range(length):
            if k % 2 == 0:
                genstring = genstring + drawBlock(x_cur, y_cur, z_left, "glass") + "\n"
                genstring = genstring + drawBlock(x_cur, y_cur, z_right, "glass") + "\n"
            else:
                genstring = (
                    genstring + drawBlock(x_cur, y_cur, z_left, blocktype) + "\n"
                )
                genstring = (
                    genstring + drawBlock(x_cur, y_cur, z_right, blocktype) + "\n"
                )
            x_cur = x_cur + 1 if x_cur >= 0 else x_cur - 1

        y_cur = y_cur + 1 if y_cur >= 0 else y_cur - 1

    # floor
    genstring = genstring + GenCuboid(
        x_back + 1 if x_back >= 0 else x_back - 1,
        y,
        z_left + 1 if z_left >= 0 else z_left - 1,
        x - 1 if x >= 0 else x + 1,
        y,
        z - 1 if x >= 0 else z + 1,
        blocktype,
    )
    # ceiling
    genstring = genstring + GenCuboid(
        x_back + 1 if x_back >= 0 else x_back - 1,
        y_cur,
        z_left + 1 if z_left >= 0 else z_left - 1,
        x - 1 if x >= 0 else x + 1,
        y_cur,
        z - 1 if x >= 0 else z + 1,
        blocktype,
    )

    return genstring


# Initialize the location coordinates
agent_x, agent_y, agent_z, agent_yaw = 19, 230, -2, 90
key_x, key_y, key_z = 7, 228, -10
door_x, door_y, door_z = -4, 228, -3
diamond_x, diamond_y, diamond_z = -18, 228, -3
start_x, start_y, start_z, width, length, height = -4, 226, -1, 6, 15, 5
holds_key = False

# relocates the agent to the starting point
def relocate_agent(agent):
    print("Agent relocated to original place")
    agent.sendCommand("tp {} {} {}".format(agent_x, agent_y, agent_z))
    agent.sendCommand("setYaw {}".format(agent_yaw))
    time.sleep(0.2)


# stop the agent
def stop_condition(agent, blocktype, sec=10):
    print("Moving to the {}".format(blocktype))

    def stop(agent, blocktype):
        done = False
        world_state = agent.peekWorldState()
        while not done:
            world_state = agent.peekWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                ob = json.loads(world_state.observations[-1].text)
                if blocktype in ob["BlocksInFront"]:
                    print("reached at {}".format(blocktype))
                    agent.sendCommand("move 0")
                    done = True

    with timeout(sec):
        try:
            stop(agent, blocktype)
            return True
        except TimeoutError:
            print("*********** Time out **************")
            agent.sendCommand("move 0")
            relocate_agent(agent)
            return False


# Get realtime location coordinates
def get_curr_loc(agent):
    loc = (agent_x, agent_y, agent_z)
    world_state = agent.peekWorldState()
    # while world_state.is_mission_running:
    #   world_state = agent.peekWorldState()
    if world_state.number_of_observations_since_last_state > 0:
        ob = json.loads(world_state.observations[-1].text)
        loc = (int(ob[u"XPos"]), int(ob[u"YPos"]), int(ob[u"ZPos"]))

    return loc


# Turns the agent towards the location given
def turn_to(agent, x1, y1, z1, x2, y2, z2):
    def get_angle(x1, y1, z1, x2, y2, z2):
        x_dist = abs(x1 - x2)
        z_dist = abs(z1 - z2)
        h = math.sqrt(x_dist ** 2 + z_dist ** 2)
        return math.degrees(z_dist / h) if h != 0 else 0

    angle = get_angle(x1, y1, z1, x2, y2, z2)
    yaw = 90 if x1 > x2 else -90
    if yaw > 0:
        angle = yaw - angle if z1 < z2 else yaw + angle
    else:
        angle = yaw + angle if z1 < z2 else yaw - angle
    agent.sendCommand("setYaw {}".format(angle))


def is_mission_running(agent):
    world_state = agent.peekWorldState()
    return world_state.is_mission_running


# Move the agent towards the key
# reward=0
def go_to_the_key(agent):
    global holds_key
    curr_x, curr_y, curr_z = get_curr_loc(agent)
    turn_to(agent, curr_x, curr_y, curr_z, key_x, key_y, key_z)
    time.sleep(0.2)
    agent.sendCommand("move 1")
    action_complete = stop_condition(agent, "tripwire_hook")
    if action_complete:
        agent.sendCommand("hotbar.2 1")
        observation = {"observation": ["holds_key"]}
        holds_key = True
    else:
        observation = []
    reward = 0
    return observation, reward, is_mission_running


# Move the agent towards the house and enter the house.
# reward = 0
def go_to_the_house(agent):
    curr_x, curr_y, curr_z = get_curr_loc(agent)
    turn_to(agent, curr_x, curr_y, curr_z, door_x, door_y, door_z)
    time.sleep(0.2)
    agent.sendCommand("move 1")
    action_complete = stop_condition(agent, "dark_oak_door")
    if action_complete:
        observation = {"observation": []}
        if holds_key:
            agent.sendCommand("tp {} {} {}".format(door_x, door_y, door_z))
            observation["observation"].append("inside_the_house")
            observation["observation"].append("holds_key")
        else:
            observation["observation"].append("nextto_closed_door")

    else:
        observation = []
    reward = 0
    return observation, reward, is_mission_running


# Move the agent towards the diamond and collect it.
# reward = 1
def go_to_the_diamonds(agent):
    curr_x, curr_y, curr_z = get_curr_loc(agent)
    turn_to(agent, curr_x, curr_y, curr_z, diamond_x, diamond_y, diamond_z)
    time.sleep(0.2)
    agent.sendCommand("move 1")
    action_complete = stop_condition(agent, "diamond_block")
    if action_complete:
        reward = 1
    else:
        reward = 0
    return [], reward, is_mission_running


missionXML = (
    """
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Collect diamonds!</Summary>
  </About>

  <ServerSection>
    <ServerInitialConditions>
            <Time>
                <StartTime>1000</StartTime>
                <AllowPassageOfTime>true</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>true</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;1;biome_1"/>
        <DrawingDecorator>
          """
    + build_house(start_x, start_y, start_z, width, length, height, "planks")
    + """
          """
    + drawLine(
        door_x, door_y - 1, door_z, door_x, door_y, door_z, "dark_oak_door", "EAST"
    )
    + """
          """
    + drawBlock(key_x, key_y - 1, key_z, "command_block")
    + """
          """
    + drawBlock(key_x, key_y, key_z, "tripwire_hook")
    + """

        </DrawingDecorator>
      <ServerQuitFromTimeUp description="" timeLimitMs="1000000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Cog Agent</Name>
    <AgentStart>
      """
    + '<Placement x="{}" y="{}" z="{}" yaw="{}"/>'.format(
        agent_x, agent_y, agent_z, agent_yaw
    )
    + """
        <Inventory>
            <InventoryItem slot="0" type="diamond_pickaxe"/>
            <InventoryItem slot="1" type="tripwire_hook"/>
        </Inventory>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ObservationFromGrid>
        <Grid name="BlocksInFront">
          <min x="-1" y="1" z="-1"/>
          <max x="1" y="1" z="1"/>
        </Grid>
      </ObservationFromGrid>
      <ObservationFromRecentCommands/>
      <ContinuousMovementCommands turnSpeedDegs="180"/>
      <AbsoluteMovementCommands />
      <InventoryCommands/>
      <RewardForTouchingBlockType>
        <Block type="diamond_block" reward="1.0"/>
      </RewardForTouchingBlockType>
    </AgentHandlers>
  </AgentSection>

</Mission>

"""
)

if __name__ == "__main__":
    """
    The following is some random heuristics to demo how you
    would use the MalmoWrapper and the environment defined
    with the xml above.
    """

    a = AtomSpace()
    set_default_atomspace(a)

    malmoWrapper = MalmoWrapper(missionXML=missionXML, validate=True)

    rw, ob, done = malmoWrapper.restart()
    total_reward = 0
    agent = malmoWrapper.agent_host
    _, rw, _ = go_to_the_key(agent)
    print(ob)
    total_reward += rw
    world_state = agent.getWorldState()
    while world_state.is_mission_running:
        world_state = agent.getWorldState()
        if world_state.number_of_rewards_since_last_state > 0:
            total_reward += world_state.rewards[-1].getValue()
            print("total reward minecraft = {}".format(total_reward))

        ob, rw, _ = go_to_the_house(agent)
        print(ob)
        total_reward += rw

        ob, rw, _ = go_to_the_diamonds(agent)
        print(ob)
        total_reward += rw

        time.sleep(2)
        print("total reward = {}".format(total_reward))
    print()
    print("Mission ended")
