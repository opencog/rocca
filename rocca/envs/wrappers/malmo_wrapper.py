import uuid
from functools import wraps
from builtins import range
import os
import sys
import time
import json

from rocca.malmo import MalmoPython

from .utils import *
from .wrapper import Wrapper

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(
        sys.stdout.fileno(), "w", 0
    )  # flush print output immediately
else:
    import functools

    print = functools.partial(print, flush=True)


class MalmoWrapper(Wrapper):
    def __init__(
        self, missionXML, validate, setup_mission=None, ip="127.0.0.1", port=10000
    ):
        super()
        self.agent_host = MalmoPython.AgentHost()
        self.clientPool = MalmoPython.ClientPool()
        self.clientPool.add(MalmoPython.ClientInfo(ip, port))
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print("ERROR:", e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)

        self.mission = MalmoPython.MissionSpec(missionXML, validate)
        self.mission_record = MalmoPython.MissionRecordSpec()
        if setup_mission is not None:
            setup_mission(self.mission)

    @classmethod
    def fromfile(cls, path, validate, setup_mission=None):
        with open(path, "r") as f:
            print("Loading mission from {}".format(path))
            missionXML = f.read()
        return cls(missionXML, validate, setup_mission=setup_mission)

    @staticmethod
    def parse_world_state(ws):
        if not ws.rewards:
            rw = 0
        else:
            rw = ws.rewards[-1].getValue()  # TODO Take all rewards if multi-agent.

        obs_list = []
        if ws.number_of_observations_since_last_state > 0:
            observations = json.loads(ws.observations[-1].text)
            for k in observations:
                if isinstance(observations[k], list):
                    obs_list.append(mk_evaluation(k, *observations[k]))
                else:
                    obs_list.append(mk_evaluation(k, observations[k]))
        else:
            obs_list = []

        return obs_list, mk_evaluation("Reward", rw), not ws.is_mission_running

    @staticmethod
    def restart_decorator(restart):
        @wraps(restart)
        def wrapper(*args, **kwargs):
            ws = restart(*args, **kwargs)
            return MalmoWrapper.parse_world_state(ws)

        return wrapper

    @staticmethod
    def step_decorator(step):
        @wraps(step)
        def wrapper(ref, action, update_callback=None):
            name, value = action.out[0], action.out[1]
            ws = step(ref, name.name + " " + value.name, update_callback)
            return MalmoWrapper.parse_world_state(ws)

        return wrapper

    @restart_decorator.__func__
    def restart(self, max_retries=3):
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(
                    self.mission,
                    self.clientPool,
                    self.mission_record,
                    0,
                    str(uuid.uuid4()),
                )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        # Loop until mission starts:
        print("Waiting for the mission to start ", end=" ")
        self.world_state = self.agent_host.getWorldState()
        while not self.world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            self.world_state = self.agent_host.getWorldState()
            for error in self.world_state.errors:
                print("Error:", error.text)
        print()
        print("Mission running ", end=" ")
        return self.world_state

    @step_decorator.__func__
    def step(self, action, update_callback=None):
        try:
            self.agent_host.sendCommand(action)
            time.sleep(0.2)
            self.world_state = self.agent_host.getWorldState()

            for error in self.world_state.errors:
                print("Error: ", error.text)

            if update_callback is not None:
                update_callback(action, self.world_state)
        except RuntimeError as e:
            print("Error sending command:", e)
        return self.world_state

    def close(self):
        # Clean up env.
        pass
