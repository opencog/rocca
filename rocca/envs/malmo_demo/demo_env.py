from past.utils import old_div
from opencog.type_constructors import *

from envs.wrappers.malmo_wrapper import MalmoWrapper
from envs.wrappers.utils import *

import time

"""This code is taken from the malmo tutorial examples."""


def Menger(xorg, yorg, zorg, size, blocktype, variant, holetype):
    # draw solid chunk
    genstring = (
        GenCuboidWithVariant(
            xorg,
            yorg,
            zorg,
            xorg + size - 1,
            yorg + size - 1,
            zorg + size - 1,
            blocktype,
            variant,
        )
        + "\n"
    )
    # now remove holes
    unit = size
    while unit >= 3:
        w = old_div(unit, 3)
        for i in range(0, size, unit):
            for j in range(0, size, unit):
                x = xorg + i
                y = yorg + j
                genstring += (
                    GenCuboid(
                        x + w,
                        y + w,
                        zorg,
                        (x + 2 * w) - 1,
                        (y + 2 * w) - 1,
                        zorg + size - 1,
                        holetype,
                    )
                    + "\n"
                )
                y = yorg + i
                z = zorg + j
                genstring += (
                    GenCuboid(
                        xorg,
                        y + w,
                        z + w,
                        xorg + size - 1,
                        (y + 2 * w) - 1,
                        (z + 2 * w) - 1,
                        holetype,
                    )
                    + "\n"
                )
                genstring += (
                    GenCuboid(
                        x + w,
                        yorg,
                        z + w,
                        (x + 2 * w) - 1,
                        yorg + size - 1,
                        (z + 2 * w) - 1,
                        holetype,
                    )
                    + "\n"
                )
        unit = w
    return genstring


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


def GenCuboidWithVariant(x1, y1, z1, x2, y2, z2, blocktype, variant):
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
        + '" variant="'
        + variant
        + '"/>'
    )


missionXML = (
    """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Hello world!</Summary>
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
                  <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                  <DrawingDecorator>
                    <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>"""
    + Menger(-40, 40, -13, 27, "stone", "smooth_granite", "air")
    + """
                    <DrawCuboid x1="-25" y1="39" z1="-2" x2="-29" y2="39" z2="2" type="lava"/>
                    <DrawCuboid x1="-26" y1="39" z1="-1" x2="-28" y2="39" z2="1" type="obsidian"/>
                    <DrawBlock x="-27" y="39" z="0" type="diamond_block"/>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="30000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="0.5" y="56.0" z="0.5" yaw="90"/>
                    <Inventory>
                        <InventoryItem slot="8" type="diamond_pickaxe"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ObservationFromGrid>
                      <Grid name="Floor3x3">
                        <min x="-1" y="-1" z="-1"/>
                        <max x="1" y="-1" z="1"/>
                      </Grid>
                  </ObservationFromGrid>
                  <ObservationFromRecentCommands/>
                  <ObservationFromHotBar/>
                  <ObservationFromDistance>
                    <Marker x="10"
                      y="56"
                      z="0.5"
                      name="Mark1"/>
                  </ObservationFromDistance>
                  <ObservationFromDiscreteCell/> 
                  <ObservationFromTurnScheduler/>
                  <RewardForReachingPosition >
                    <Marker x="-5"
                      y="56"
                      z="0.5"
                      reward="10000"
                      tolerance="3"
                      oneshot="false"
                      distribution="MalmoTutorialBot:1"/>
                  </RewardForReachingPosition>
                  <RewardForTouchingBlockType>
                    <Block reward="10000" type="air" distribution="MalmoTutorialBot:1"/>
                  </RewardForTouchingBlockType>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <InventoryCommands/>
                  <AgentQuitFromTouchingBlockType>
                      <Block type="diamond_block" />
                  </AgentQuitFromTouchingBlockType>
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

    malmoWrapper = MalmoWrapper(missionXML=missionXML, validate=True)

    def stp_callback(action, ws):
        pass  # you can do something here.

    rw, ob, done = malmoWrapper.restart()

    malmoWrapper.step(mk_action("hotbar.9", 1))  # Press the hotbar key
    malmoWrapper.step(
        mk_action("hotbar.9", 0)
    )  # Release hotbar key - agent should now be holding diamond_pickaxe
    time.sleep(1)  # Wait a second until we are looking in roughly the right direction
    malmoWrapper.step(mk_action("move", 0.8))  # And start running...
    malmoWrapper.step(mk_action("attack", 1))
    malmoWrapper.step(mk_action("jump", 1))
    while not done:
        print(".", end="")
        time.sleep(0.2)
        rw, ob, done = malmoWrapper.step(
            mk_action("jump", 1), update_callback=stp_callback
        )
        print("Reward : \n", rw)
        print("Observation : \n", ob)
        print()

    print()
    print("Mission ended")
