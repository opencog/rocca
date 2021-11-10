import os
import time

from fastcore.basics import gen

from rocca.envs.wrappers.malmo_wrapper import MalmoWrapper
from rocca.envs.wrappers.utils import *
from builtins import range
from past.utils import old_div
import os
import sys
import time
import json
import random
import math

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

def drawBlock(x, y, z, type):
    return """<DrawBlock x='{0}' y='{1}' z='{2}' type='{3}'/>""".format(x, y, z, type)

def GenCuboid(x1, y1, z1, x2, y2, z2, blocktype):
    return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

def build_house(x, y, z, width, length, height, blocktype):
    x_front = x
    x_back = x + length if x >= 0 else x-length
    z_left = z + width if z >= 0 else z-width
    z_right = z

    genstring = ""
    y_cur = y
    for i in range(height):
      z_cur = z
      x_cur = x
      for j in range(width):
        genstring = genstring + drawBlock(x_front, y_cur, z_cur, "glass") + "\n"
        genstring = genstring + drawBlock(x_back, y_cur, z_cur, blocktype) + "\n"
        z_cur = z_cur + 1 if z_cur >= 0 else z_cur-1

      for k in range(length):
        genstring = genstring + drawBlock(x_cur, y_cur, z_left, blocktype) + "\n"
        genstring = genstring + drawBlock(x_cur, y_cur, z_right, blocktype) + "\n"
        x_cur = x_cur + 1 if x_cur >= 0 else x_cur-1

      y_cur = y_cur + 1 if y_cur >= 0 else y_cur-1
      
    # floor
    genstring = genstring + GenCuboid(x_back + 1 if x_back >= 0 else x_back - 1, y, z_left + 1 if z_left >= 0 else z_left - 1, x - 1 if x >= 0 else x + 1, y, z - 1 if x >= 0 else z + 1, blocktype )
    # ceiling
    genstring = genstring + GenCuboid(x_back + 1 if x_back >= 0 else x_back - 1, y_cur, z_left + 1 if z_left >= 0 else z_left - 1, x - 1 if x >= 0 else x + 1, y_cur, z - 1 if x >= 0 else z + 1, blocktype )
    
    return genstring


agent_x, agent_y, agent_z, agent_yaw = 9, 230, -2, 90

missionXML = '''
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
          '''+build_house(-4, 226, -1,6,6,5,"planks")+'''
          <DrawBlock x="-8" y="228" z="-3" type="diamond_block"/>


          <DrawLine x1="-4" y1="227" z1="-3" x2="-4" y2="228" z2="-3" type="dark_oak_door" face="EAST"/>

          <DrawBlock x="-4" y="227" z="-15" type="command_block"/>
          <DrawBlock x="-4" y="228" z="-15" type="tripwire_hook"/>


        </DrawingDecorator>
      <ServerQuitFromTimeUp description="" timeLimitMs="1000000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Cog Agent</Name>
    <AgentStart>
      '''+'<Placement x="{}" y="{}" z="{}" yaw="{}"/>'.format(agent_x, agent_y, agent_z, agent_yaw)+'''
        <Inventory>
            <InventoryItem slot="0" type="diamond_pickaxe"/>
            <InventoryItem slot="1" type="tripwire_hook"/>
        </Inventory>
    </AgentStart>
    <AgentHandlers>
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
        <Block type="diamond_block" reward="1" />
      </RewardForTouchingBlockType>
      <RewardForCollectingItem>
          <Item reward="1" type="diamond_block" />
      </RewardForCollectingItem>
    </AgentHandlers>
  </AgentSection>

</Mission>

'''

# Moves the agent to the position given 
def turn_to(agent,x1, y1, z1, yaw, x2, y2, z2):
  x_dist = abs(x1) + abs(x2)
  z_dist = abs(z1) + abs(z2)
  h = math.sqrt(x_dist**2 + z_dist**2)
  angle = yaw + math.degrees(z_dist/h)
  agent.step(mk_action("setYaw",angle))

# Agent goes to the key
# reward=0 
def go_to_the_key(agent, x, y, z):
  turn_to(agent,agent_x, agent_y, agent_z, agent_yaw, x, y, z)
  agent.step(mk_action("move", 1))


# Agent stops and grab the key
# observations: agent holds the key and reward=0 
def get_the_key(agent):
  global agent_z
  agent_z = -15
  agent.step(mk_action("move", 0))
  agent.step(mk_action("hotbar.2", 1))

# Agent goes to the house. The agent should have the key
# observations:
# - Agent is next to closed door
# - Agent holds the key
# - reward = 0
def go_to_the_house(agent, x, y, z):
  agent.step(mk_action("tpx", agent_x))
  time.sleep(0.2)
  turn_to(agent, x, y, z)
  agent.step(mk_action("move", 1))

# Agent opens the house.
# observations:
# - Agent next to opened door
# - reward = 0
def open_the_door(agent):
  agent.step(mk_action("move", 0))
  agent.step(mk_action("attack", 1))

# Agent goes to the diamond inside the house.
# reward = 0
def go_to_the_diamond():
  return True

# Agent collects the diamond inside the house
# reward = 1
def collect_diamond(agent):
  agent.step(mk_action("hotbar.1", 1))
  rw, ob, _ = agent.step(mk_action("attack", 1))
  return rw, ob

if __name__ == "__main__":
    """
    The following is some random heuristics to demo how you
    would use the MalmoWrapper and the environment defined
    with the xml above.
    """

    a = AtomSpace()
    set_default_atomspace(a)

    malmoWrapper = MalmoWrapper(missionXML=missionXML, validate=True)

    def stp_callback(action, ws):
        pass  # you can do something here.

    rw, ob, done = malmoWrapper.restart()
    total_reward = 0
    # main loop:
    time.sleep(0.5)
    malmoWrapper.step(mk_action("hotbar.1", 0))
    malmoWrapper.step(mk_action("hotbar.2", 0))

    go_to_the_key(malmoWrapper, -4, 228, -15)

    while not done :
        print(".", end="")
        time.sleep(0.5)
        agent = malmoWrapper.agent_host
        world_state = malmoWrapper.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
          ob = world_state.observations[-1].text

          print(ob)
          if "tripwire_hook" in ob:
            malmoWrapper.step(mk_action("move", 0))
            malmoWrapper.step(mk_action("hotbar.2", 1))
            malmoWrapper.step(mk_action("tpx", agent_x))

            malmoWrapper.step(mk_action("setYaw", 45))
            malmoWrapper.step(mk_action("move", 1))

          if "dark_oak_door" in ob:
            malmoWrapper.step(mk_action("move", 0))
            malmoWrapper.step(mk_action("setYaw", 90))
            malmoWrapper.step(mk_action("attack", 1))
            malmoWrapper.step(mk_action("move", 1))

          if "diamond_block" in ob:
            malmoWrapper.step(mk_action("move", 0))
            collect_diamond(malmoWrapper)
            
          if world_state.number_of_rewards_since_last_state > 0:
            malmoWrapper.step(mk_action("attack", 0))
            reward = world_state.rewards[-1].getValue()
            print("Reward: " + str(reward))
            total_reward += reward
            malmoWrapper.step(mk_action("tp", "{} {} {}".format(agent_x, agent_y, agent_z)))

    print()
    print("Mission ended")
