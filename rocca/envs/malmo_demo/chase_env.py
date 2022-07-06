import os
import time

from rocca.envs.wrappers.malmo_wrapper import MalmoWrapper
from rocca.envs.wrappers.utils import *

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


mission_xml = (
    """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
            <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" 
                                forceReset="true"/>
            <DrawingDecorator>
                """
    + drawRewards(228, 2000, "diamond_block")
    + """
            </DrawingDecorator>
            <ServerQuitWhenAnyAgentFinishes description=""/>
        </ServerHandlers>
    </ServerSection>
    <AgentSection mode="Survival">
        <Name>ChaseBot</Name>
        <AgentStart>
            <Placement x="0" y="230" z="0" yaw="90"/>
            <Inventory>
                <InventoryItem slot="8" type="diamond_pickaxe"/>
                <InventoryItem slot="7" type="golden_apple" quantity="10"/>
                <InventoryItem slot="6" type="golden_apple" quantity="10"/>
                <InventoryItem slot="5" type="golden_apple" quantity="10"/>
                <InventoryItem slot="4" type="golden_apple" quantity="10"/>
                <InventoryItem slot="3" type="golden_apple" quantity="10"/>
            </Inventory>
        </AgentStart>
        <AgentHandlers>
            <ObservationFromFullStats/>
            <ObservationFromGrid>
              <Grid name="BlocksInFront">
                <min x="-7" y="1" z="0"/>
                <max x="0" y="1" z="0"/>
              </Grid>
            </ObservationFromGrid>
            <ObservationFromRecentCommands/>
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
</Mission>"""
)


if __name__ == "__main__":
    """
    The following is some random heuristics to demo how you
    would use the MalmoWrapper and the environment defined
    with the xml above.
    """

    a = AtomSpace()
    set_default_atomspace(a)

    malmoWrapper = MalmoWrapper(missionXML=mission_xml, validate=True)

    def stp_callback(action, ws):
        pass  # you can do something here.

    rw, ob, done = malmoWrapper.restart()

    malmoWrapper.step(mk_action("hotbar.9", 1))  # Press the hotbar key
    malmoWrapper.step(
        mk_action("hotbar.9", 0)
    )  # Release hotbar key - agent should now be holding diamond_pickaxe

    # Wait a second until we are looking in roughly the right direction
    time.sleep(1)

    # tilt camera
    # malmoWrapper.step(mk_action("pitch", 0.3))
    # time.sleep(0.5)
    # malmoWrapper.step(mk_action("pitch", 0))

    malmoWrapper.step(mk_action("move", 0.5))

    # malmoWrapper.step(mk_action("tpz", -1.5))
    malmoWrapper.step(mk_action("tpz", 2.5))
    while not done:
        print(".", end="")
        time.sleep(0.2)
        rw, ob, done = malmoWrapper.step(
            mk_action("attack", 1), update_callback=stp_callback
        )

        print("Reward : \n", rw)
        print("Observation : \n", ob)
        print()

    print()
    print("Mission ended")
