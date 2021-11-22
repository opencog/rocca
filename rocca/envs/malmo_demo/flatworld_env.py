import os
import time

from rocca.envs.wrappers.malmo_wrapper import MalmoWrapper
from rocca.envs.wrappers.utils import *
from builtins import range
import json
import random
import math


def drawBlock(x, y, z, type):
    return """<DrawBlock x='{0}' y='{1}' z='{2}' type='{3}'/>""".format(x, y, z, type)

def GenCuboid(x1, y1, z1, x2, y2, z2, blocktype):
    return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

def drawLine(x1, y1, z1, x2, y2, z2, blocktype, face):
    return '<DrawLine x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '" face="' + face +'"/>'

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

agent_x, agent_y, agent_z, agent_yaw = 19, 230, -2, 90
key_x, key_y, key_z = 7, 228, -10
door_x, door_y, door_z = -4, 228, -3
diamond_x, diamond_y, diamond_z = -8, 228, -3 

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
          '''+drawBlock(diamond_x, diamond_y, diamond_z, "diamond_block")+'''
          '''+drawLine(door_x, door_y-1, door_z, door_x, door_y, door_z, "dark_oak_door", "EAST")+'''
          '''+drawBlock(key_x, key_y-1, key_z, "command_block")+'''
          '''+drawBlock(key_x, key_y, key_z, "tripwire_hook")+'''

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
def get_angle(x1, y1, z1, x2, y2, z2):
  x_dist = abs(x1 - x2)
  z_dist = abs(z1 - z2)
  h = math.sqrt(x_dist**2 + z_dist**2)
  return math.degrees(z_dist/h) if h != 0 else 0
 
# Turn the agent towards the location given 
def turn_to(agent, x1, y1, z1, yaw, x2, y2, z2):
  angle = get_angle(x1, y1, z1, x2, y2, z2)
  yaw = 90 if x1 > x2 else -90
  if yaw > 0:
      angle = yaw - angle if z1 < z2 else yaw + angle
  else:
      angle = yaw + angle if z1 < z2 else yaw - angle
  agent.sendCommand("setYaw {}".format(angle))

# Agent goes to the key
# reward=0 
def go_to_the_key(agent):
  global agent_x, agent_y, agent_z
  turn_to(agent,agent_x, agent_y, agent_z, agent_yaw, key_x, key_y, key_z)
  time.sleep(0.2)
  agent.sendCommand("move 1")
  stop_condition(agent, 'tripwire_hook')
  agent_x, agent_y, agent_z = key_x, key_y, key_z

# Agent stops and grab the key
# observations: agent holds the key and reward=0 
def get_the_key(agent):
  agent.sendCommand("move 0")
  agent.sendCommand("hotbar.2 1")

# Agent goes to the house. The agent should have the key
# observations:
# - Agent is next to closed door
# - Agent holds the key
# - reward = 0
def go_to_the_house(agent):
  global agent_x, agent_y, agent_z
  turn_to(agent,agent_x, agent_y, agent_z, agent_yaw, door_x, door_y, door_z)
  time.sleep(0.2)
  agent.sendCommand("move 1")
  stop_condition(agent, 'dark_oak_door')
  agent_x, agent_y, agent_z = door_x, door_y, door_z

# Agent opens the house.
# observations:
# - Agent next to opened door
# - reward = 0
def open_the_door(agent):
  agent.sendCommand("move 0")
  agent.sendCommand("attack 1")

# Agent goes to the diamond inside the house.
# reward = 0
def go_to_the_diamonds(agent):
  global agent_x, agent_y, agent_z
  turn_to(agent,agent_x, agent_y, agent_z, agent_yaw, diamond_x, diamond_y, diamond_z)
  time.sleep(0.2)
  agent.sendCommand("move 1")
  stop_condition(agent, "diamond_block")
  agent_x, agent_y, agent_z = diamond_x, diamond_y, diamond_z

def stop_condition(agent, block_type):
  done = False
  while not done:
    world_state = agent.getWorldState()
    if world_state.number_of_observations_since_last_state > 0:
      ob = world_state.observations[-1].text
      if block_type in ob:
        agent.sendCommand("move 0")
        done = True

# Agent collects the diamond inside the house
# reward = 1
def collect_diamonds(agent):
  agent.sendCommand("hotbar.1 1")
  agent.sendCommand("attack 1")


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

    agent = malmoWrapper.agent_host

    go_to_the_key(agent)

    while not done:
        print(".", end="")
        time.sleep(0.5)
        world_state = malmoWrapper.agent_host.getWorldState()

        if world_state.number_of_observations_since_last_state > 0:
          ob = world_state.observations[-1].text
          go_to_the_house(agent)
          get_the_key(agent)
          open_the_door(agent)
          collect_diamonds(agent)
          malmoWrapper.step(mk_action("tp", "{} {} {}".format(19, 230, -2)))
          done = True
    print()
    print("Mission ended")
