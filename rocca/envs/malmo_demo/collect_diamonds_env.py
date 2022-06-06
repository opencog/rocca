import os
import time

from rocca.envs.wrappers.malmo_wrapper import MalmoWrapper
from rocca.envs.wrappers.utils import *
from builtins import range
import json
import math
import signal
from contextlib import contextmanager
import random


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


def drawBlock(x, y, z, type, variant=False, face=False):
    if variant:
        return """<DrawBlock x='{0}' y='{1}' z='{2}' type='{3}' variant='{4}' face='{5}'/>""".format(
            x, y, z, type, variant, face
        )
    else:
        return """<DrawBlock x='{0}' y='{1}' z='{2}' type='{3}' />""".format(
            x, y, z, type, variant
        )


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


def drawItem(x, y, z, item_type):
    return (
        '<DrawItem x="'
        + str(x)
        + '" y="'
        + str(y)
        + '" z="'
        + str(z)
        + '" type="'
        + item_type
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


def draw_fence(x, y, z):
    fence = ""
    xy_coord = [(-x, z), (x, z), (x, -z), (-x, -z)]
    for i in range(4):
        fence += drawLine(
            x1=xy_coord[i][0],
            y1=y,
            z1=xy_coord[i][1],
            x2=xy_coord[i + 1][0] if i < 3 else xy_coord[0][0],
            y2=y,
            z2=xy_coord[i + 1][1] if i < 3 else xy_coord[0][1],
            blocktype="fence",
            face="WEST",
        )
    return fence


def build_house(x, y, z, width, length, height, blocktype):
    x_front = x
    x_back = x + length if x >= 0 else x - length
    z_left = z + width if z >= 0 else z - width
    z_right = z

    genstring = ""
    y_cur = y
    for i in range(height + 1):
        z_cur = z
        x_cur = x
        for j in range(width + 1):
            genstring = genstring + drawBlock(x_front, y_cur, z_cur, "glass") + "\n"
            genstring = genstring + drawBlock(x_back, y_cur, z_cur, blocktype) + "\n"
            if j > 0 and j < width:
                genstring = (
                    genstring
                    + drawBlock(x_back + 1, y_cur, z_cur, "diamond_ore")
                    + "\n"
                )
            z_cur = z_cur + 1 if z_cur >= 0 else z_cur - 1

        for k in range(length):
            if k % 2 != 0:
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
        -15,
        y,
        15,
        15,
        y,
        -15,
        "obsidian",
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
    genstring = genstring + GenCuboid(
        x_back,
        y_cur + 1,
        z_left,
        x,
        y_cur + 1,
        z,
        blocktype,
    )
    genstring = genstring + GenCuboid(
        x_back - 1 if x_back >= 0 else x_back + 1,
        y_cur + 2,
        z_left - 1 if z_left >= 0 else z_left + 1,
        x + 1 if x >= 0 else x - 1,
        y_cur + 2,
        z + 1 if x >= 0 else z - 1,
        blocktype,
    )
    genstring = genstring + GenCuboid(
        x_back,
        y_cur,
        z_left,
        x,
        y_cur,
        z,
        "air",
    )
    genstring = genstring + GenCuboid(
        x_back - 1 if x_back >= 0 else x_back + 1,
        y_cur + 1,
        z_left - 1 if z_left >= 0 else z_left + 1,
        x + 1 if x >= 0 else x - 1,
        y_cur + 1,
        z + 1 if x >= 0 else z - 1,
        "air",
    )
    genstring = genstring + GenCuboid(
        x_back - 2 if x_back >= 0 else x_back + 2,
        y_cur + 2,
        z_left - 2 if z_left >= 0 else z_left + 2,
        x + 2 if x >= 0 else x - 2,
        y_cur + 2,
        z + 2 if x >= 0 else z - 2,
        "glass",
    )

    return genstring


# Initialize the location coordinates
agent_x, agent_y, agent_z, agent_yaw = 13, 230, -2, 90
key_x, key_y, key_z = 7, 228, -10
door_x, door_y, door_z = -4, 228, -3
exit_x, exit_y, exit_z = -9, 228, -1
diamond_x, diamond_y, diamond_z = -18, 228, -3
start_x, start_y, start_z, width, length, height = -4, 226, -1, 6, 9, 4
hold_key = False
fence_max = 15

place_count = 0
pitch_adjusted = False
total_reward = 0


def adjust_pitch(agent):
    global pitch_adjusted
    if not pitch_adjusted:
        agent.sendCommand("pitch 0.5")
        time.sleep(0.5)
        agent.sendCommand("pitch 0")
        pitch_adjusted = True


# relocates the agent to the starting point
def relocate_agent(agent, msg=""):
    print(msg)
    agent.sendCommand("tp {} {} {}".format(agent_x, agent_y, agent_z))
    if not hold_key:
        agent.sendCommand("hotbar.4 1")
    agent.sendCommand("setYaw {}".format(agent_yaw))
    time.sleep(0.2)


blocks = {
    "tripwire_hook": "key",
    "dark_oak_door": "house",
    "diamond_block": "diamonds",
    "planks": "planks",
    "diamond_ore": "collect diamonds",
}

# stop the agent
def stop_condition(agent, blocktype, sec=10):
    print("\nMoving to {} \n".format(blocks[blocktype]))

    def stop(agent, blocktype):
        done = False
        world_state = agent.peekWorldState()
        while not done:
            world_state = agent.peekWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                ob = json.loads(world_state.observations[-1].text)
                if blocktype in ob["BlocksInFront"]:
                    agent.sendCommand("move 0")
                    done = True

    with timeout(sec):
        try:
            stop(agent, blocktype)
            return True
        except TimeoutError:
            agent.sendCommand("move 0")
            relocate_agent(agent, msg="*** RELOCATE: Unable to reach the target ***")
            return False


# Get realtime location coordinates
def get_curr_loc(agent):
    loc = (agent_x, agent_y, agent_z)
    world_state = agent.peekWorldState()
    while world_state.is_mission_running:
        world_state = agent.peekWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            ob = json.loads(world_state.observations[-1].text)
            yaw = int(ob["Yaw"])
            loc = (int(ob[u"XPos"]), int(ob[u"YPos"]), int(ob[u"ZPos"]), yaw)
            break
    return loc


# Turns the agent towards the location given
def turn_to(agent, x1, y1, z1, x2, y2, z2, curr_yaw=False):
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
    if curr_yaw:
        if curr_yaw < angle:
            agent.sendCommand("turn 0.5")
        else:
            agent.sendCommand("turn -0.5")
        while is_mission_running:
            world_state = agent.peekWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                ob = json.loads(world_state.observations[-1].text)
                yaw = int(ob["Yaw"])
                if abs(yaw - angle) < 5:
                    agent.sendCommand("turn 0")
                    agent.sendCommand("setYaw {}".format(angle))
                    break
    else:
        agent.sendCommand("setYaw {}".format(angle))


def is_mission_running(agent):
    world_state = agent.peekWorldState()
    return world_state.is_mission_running


def inside_house(x, z):
    return (
        True
        if (x in range(start_x, start_x - length, -1))
        and (z in range(start_z, start_z - width, -1))
        else False
    )


def get_txt(obs):
    if obs:
        return ",".join(
            ["{}({},{})".format(ob, obs[ob][0], obs[ob][1]) for ob in obs.keys()]
        )
    else:
        return ""


# Move the agent towards the key
# reward=0
def go_to_key(agent):
    adjust_pitch(agent)
    global hold_key
    reward = 0
    observation = {}
    curr_x, curr_y, curr_z, curr_yaw = get_curr_loc(agent)
    if inside_house(curr_x, curr_z):
        observation["inside"] = ["self", "house"]
        print("Observation: {}".format(get_txt(observation)))
        return observation, reward, is_mission_running
    else:
        observation["outside"] = ["self", "house"]
    if hold_key:
        observation["hold"] = ["self", "key"]
        print("Observation: {}".format(get_txt(observation)))
        return observation, reward, is_mission_running
    turn_to(agent, curr_x, curr_y, curr_z, key_x, key_y, key_z, curr_yaw=curr_yaw)
    time.sleep(0.2)
    agent.sendCommand("move 1")
    action_complete = stop_condition(agent, "tripwire_hook")
    if action_complete:
        observation["hold"] = ["self", "key"]
        agent.sendCommand("hotbar.2 1")
        hold_key = True
    else:
        observation = []
    print("Observation: {}".format(get_txt(observation)))
    return observation, reward, is_mission_running


# Move the agent towards the house and enter the house.
# reward = 0
def go_to_house(agent):
    adjust_pitch(agent)
    global hold_key
    reward = 0
    observation = {}
    curr_x, curr_y, curr_z, curr_yaw = get_curr_loc(agent)
    if inside_house(curr_x, curr_z):
        observation["inside"] = ["self", "house"]
        print("Observation: {}".format(get_txt(observation)))
        return observation, reward, is_mission_running
    turn_to(
        agent, curr_x, curr_y, curr_z, door_x + 2, door_y, door_z, curr_yaw=curr_yaw
    )
    time.sleep(0.2)
    agent.sendCommand("move 1")
    action_complete = stop_condition(agent, "dark_oak_door")
    if action_complete:
        if hold_key:
            agent.sendCommand("move 1")
            agent.sendCommand("tp {} {} {}".format(door_x, door_y, door_z))
            time.sleep(0.2)
            agent.sendCommand("move 0")
            observation["inside"] = ["self", "house"]
            agent.sendCommand("hotbar.1 1")
            hold_key = False
        else:
            observation["outside"] = ["self", "house"]
            observation["nextto"] = ["self", "closed_door"]
    reward = 0
    print("Observation: {}".format(get_txt(observation)))
    return observation, reward, is_mission_running


def place_diamonds(agent):
    global place_count
    curr_x, curr_y, curr_z, curr_yaw = get_curr_loc(agent)
    turn_to(
        agent,
        curr_x,
        curr_y,
        curr_z,
        (exit_x + 2) + (2 * place_count),
        exit_y,
        fence_max,
    )
    agent.sendCommand("move 1")
    time.sleep(2 - (place_count * 0.01))
    placed = False
    while not placed:
        world_state = agent.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            # Use the line-of-sight observation to determine where to place:
            if u"LineOfSight" in ob:
                agent.sendCommand("turn {}".format(random.choice([0.05, -0.05])))
                los = ob[u"LineOfSight"]
                x = int(math.floor(los["x"]))
                z = int(math.floor(los["z"]))
                if los["inRange"]:
                    agent.sendCommand("move 0")
                    time.sleep(2)
                    agent.sendCommand("use")
                    agent.sendCommand("turn 0")
                    agent.sendCommand("hotbar.5 1")
                    time.sleep(2)
                    place_count += 1
                    placed = True


def exit(agent):
    curr_x, curr_y, curr_z, curr_yaw = get_curr_loc(agent)
    turn_to(agent, curr_x, curr_y, curr_z, exit_x, exit_y, exit_z)
    agent.sendCommand("move 1")
    agent.sendCommand("use 1")
    time.sleep(0.2)
    while is_mission_running(agent):
        curr_x, curr_y, curr_z, _ = get_curr_loc(agent)
        if not inside_house(curr_x, curr_z):
            agent.sendCommand("move 0")
            agent.sendCommand("tp {} {} {}".format(exit_x + 0.5, curr_y, exit_z + 1))
            time.sleep(0.1)
            curr_x, curr_y, curr_z, _ = get_curr_loc(agent)
            turn_to(agent, curr_x, curr_y, curr_z, exit_x, exit_y, exit_z)
            agent.sendCommand("use 1")
            time.sleep(0.05)
            agent.sendCommand("use 0")
            break
    agent.sendCommand("hotbar.3 1")
    place_diamonds(agent)


# Move the agent towards the diamond and collect it.
# reward = 1
def go_to_diamonds(agent):
    global total_reward
    adjust_pitch(agent)
    # agent.sendCommand("chat Collect diamonds")
    observation = {}
    curr_x, curr_y, curr_z, curr_yaw = get_curr_loc(agent)
    turn_to(
        agent,
        curr_x,
        curr_y,
        curr_z,
        diamond_x,
        diamond_y,
        diamond_z,
        curr_yaw=curr_yaw,
    )
    time.sleep(0.2)
    agent.sendCommand("move 1")
    action_complete = stop_condition(agent, "diamond_ore")
    if action_complete:
        agent.sendCommand("attack 1")
        time.sleep(0.2)
        agent.sendCommand("attack 0")
        time.sleep(0.2)
        reward = 1
        total_reward += 1
        exit(agent)
        relocate_agent(agent)
        time.sleep(0.3)
        observation["outside"] = ["self", "house"]
    else:
        reward = 0
    if not inside_house(curr_x, curr_z):
        observation["outside"] = ["self", "house"]
    if hold_key:
        observation["hold"] = ["self", "key"]
    if reward > 0:
        agent.sendCommand("chat Reward = {}".format(reward))
        agent.sendCommand("chat: Total Reward of {} diamonds!".format(total_reward))
        print("Reward = {}".format(reward))
        print("Total Reward of {} diamonds!".format(total_reward))

    else:
        print("Observation: {}".format(get_txt(observation)))

    return observation, reward, is_mission_running


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
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>true</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;1;village,decoration"/>
        <DrawingDecorator>
          """
    + build_house(start_x, start_y, start_z, width, length, height, "brick_block")
    + drawBlock(diamond_x + 5, diamond_y, diamond_z - 1, "diamond_block")
    + drawBlock(
        door_x, door_y - 1, door_z, "dark_oak_door", variant="lower", face="WEST"
    )
    + drawBlock(door_x, door_y, door_z, "dark_oak_door", variant="upper", face="WEST")
    + drawBlock(exit_x, exit_y, exit_z, "dark_oak_door", variant="upper", face="NORTH")
    + drawBlock(
        exit_x, exit_y - 1, exit_z, "dark_oak_door", variant="lower", face="NORTH"
    )
    + drawBlock(key_x, key_y - 1, key_z, "stone")
    + drawBlock(key_x, key_y, key_z, "tripwire_hook")
    + draw_fence(fence_max, start_y + 1, fence_max)
    + """

        </DrawingDecorator>
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
            <InventoryBlock slot="2" type="diamond_block" quantity="63"/>
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
      <ObservationFromRay/>
      <ObservationFromRecentCommands/>
      <ContinuousMovementCommands turnSpeedDegs="80"/>
      <AbsoluteMovementCommands />
      <DiscreteMovementCommands />
      <InventoryCommands/>
      <ChatCommands/>
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
    world_state = agent.getWorldState()
    # relocate_agent(agent)
    adjust_pitch(agent)
    while world_state.is_mission_running:
        # curr_x, curr_y, curr_z = get_curr_loc(agent)
        # print(curr_x, curr_y, curr_z)
        # nb = input("Enter command: ")

        # nx, ny, nz = nb.split(",")
        # turn_to(agent, curr_x, curr_y, curr_z, int(nx), int(ny), int(nz))

        #     agent.sendCommand(nb)

        world_state = agent.getWorldState()
        if world_state.number_of_rewards_since_last_state > 0:
            total_reward += world_state.rewards[-1].getValue()
            print("total reward minecraft = {}".format(total_reward))
        ob, rw, _ = go_to_key(agent)
        print("Observation: {}, reward: {}".format(ob, rw))

        ob, rw, _ = go_to_house(agent)
        print("Observation: {}, reward: {}".format(ob, rw))

        ob, rw, _ = go_to_diamonds(agent)
        print("Observation: {}, reward: {}".format(ob, rw))

        time.sleep(2)
        print("total reward = {}".format(total_reward))
    print()
    print("Mission ended")
