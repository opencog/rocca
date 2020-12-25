import os
import time

from envs.wrappers.malmo_wrapper import MalmoWrapper
from envs.wrappers.utils import *

DIR_NAME = os.path.dirname(__file__)
mission_file = os.path.join(DIR_NAME, "chase_mission.xml")


def drawBlock(x, y, z, type):
    return """<DrawBlock x='{0}' y='{1}' z='{2}' type='{3}'/>""".format(x, y, z, type)


def drawRewards(y, count=1000, type="diamond_block"):
    xml = ""
    for x in range(-2, -1 * count, -5):
        z = 2 if x % 2 == 0 else -2
        xml += drawBlock(x, y, z, type)
    return xml


mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns:xsi="http://www.w3.org/2001/XMLSchemainstance" xmlns="http://ProjectMalmo.microsoft.com"
         xsi:schemaLocation="http://ProjectMalmo.microsoft.com Mission.xsd">
    <About>
        <Summary/>
    </About>
    <ServerSection>
        <ServerInitialConditions>
            <Time>
                <StartTime>1000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
        </ServerInitialConditions>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
            <DrawingDecorator>
                ''' + drawRewards(228, 1000, "diamond_block") + '''
            </DrawingDecorator>
            <ServerQuitFromTimeUp description="" timeLimitMs="120000"/>
            <ServerQuitWhenAnyAgentFinishes description=""/>
        </ServerHandlers>
    </ServerSection>
    <AgentSection mode="Survival">
        <Name>ChaseBot</Name>
        <AgentStart>
            <Placement x="0" y="230" z="0" yaw="90"/>
            <Inventory>
                <InventoryItem slot="8" type="diamond_pickaxe"/>
            </Inventory>
        </AgentStart>
        <AgentHandlers>
            <ObservationFromGrid>
              <Grid name="BlocksInFront">
                <min x="-5" y="1" z="0"/>
                <max x="0" y="1" z="0"/>
              </Grid>
            </ObservationFromGrid>
            <ContinuousMovementCommands turnSpeedDegs="180"/>
            <AbsoluteMovementCommands />
            <InventoryCommands/>
            <RewardForCollectingItem>
                <Item type="diamond_block"
                      reward="1"
                      distribution="ChaseBot:1"></Item>
            </RewardForCollectingItem>
        </AgentHandlers>
    </AgentSection>
</Mission>'''
