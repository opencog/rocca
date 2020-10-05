from .wrapper import Wrapper
from functools import wraps
from builtins import range

from malmo import MalmoPython
import os
import sys
import time


class MalmoWrapper(Wrapper):
    def __init__(self, missionXML, validate, setup_mission=None):
        super()
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)

        self.mission = MalmoPython.MissionSpec(missionXML, validate)
        self.mission_record = MalmoPython.MissionRecordSpec()
        if (setup_mission is not None):
            setup_mission(self.mission)

    @classmethod
    def fromfile(cls, path, validate, setup_mission=None):
        with open(path, 'r') as f:
            print("Loading mission from {}".format(path))
            missionXML = f.read()
        return cls(missionXML, validate, setup_mission=setup_mission)

    @staticmethod
    def restart_decorator(restart):
        @wraps(restart)
        def wrapper(*args, **kwargs):
            return restart(*args, **kwargs)

        return wrapper

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(*args, **kwargs):
            return step(*args, **kwargs)

        return wrapper

    @restart_decorator.__func__
    def restart(self, max_retries=3):
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(self.mission, self.mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        # Loop until mission starts:
        print("Waiting for the mission to start ", end=' ')
        self.world_state = self.agent_host.getWorldState()
        while not self.world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            self.world_state = self.agent_host.getWorldState()
            for error in self.world_state.errors:
                print("Error:", error.text)
        print()
        print("Mission running ", end=' ')
        return self.world_state

    @step_decorator.__func__
    def step(self, action, update_callback=None):
        try:
            self.agent_host.sendCommand(action)
            time.sleep(0.2)
            self.world_state = self.agent_host.getWorldState()
            if (update_callback is not None):
                update_callback(action, self.world_state)
        except RuntimeError as e:
            print("Error sending command:", e)
        return self.world_state
