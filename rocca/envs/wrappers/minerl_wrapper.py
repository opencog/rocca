from collections import defaultdict
from typing import *

from gym import Env
from gym.spaces import Space
from opencog.atomspace import Atom, AtomSpace

from .utils import *
from .gym_wrapper import GymWrapper


def minerl_single_action(env: Union[Env, GymWrapper], action: Atom) -> List[Atom]:
    """Insert a single action into a no-op

    env: Gym environment or a wrapped gym environment.
    action: action of the form `Execution (Schema name) args`
    """

    noop = env.action_space.noop()
    actions = [mk_action(k, noop[k]) for k in noop if k != action.out[0].name]
    actions.append(action)

    return actions


class MineRLWrapper(GymWrapper):
    def __init__(self, env: Env, atomspace: AtomSpace, action_names=[]):
        super().__init__(env, atomspace, action_names)
        self.last_compass_angle = None

    def transform_percept(self, label: str, *args) -> List[Atom]:
        """Firstly, the MineRL environment gives us a different floating point
        reward number every step. This function converts it into +1 or -1 so that
        the pattern miner can find frequent patterns involving reward.

        Secondly, MineRL gives us a 2D image of the agent's view within Minecraft,
        but this function gives the agent the average location of each approximate
        color on screen. The colors are put in bins of 20 pixel brightness values,
        and then we record the average location of the color bin on screen. Note
        that this function doesn't divide the screen into blobs of one color; it
        may find the average of multiple blobs of one color. Note that the pattern
        miner will still have difficulty with this feature so it's a work in
        progress.

        Thirdly, MineRL gives us the angle between the agent and the goal (compassAngle).
        This function creates a boolean predicate for whether the angle has got closer
        or not."""
        if label == "pov":
            colors = defaultdict(list)

            for y in range(0, 64):
                for x in range(0, 64):
                    color = args[y][x]
                    rounded_color = tuple([subpixel // 25 * 25 for subpixel in color])
                    colors[rounded_color].append((x, y))

            observation: List[Atom] = []
            for (color, locations) in colors.items():
                total_x = total_y = 0
                for (x, y) in locations:
                    total_x += x
                    total_y += y
                observation.append(
                    AtLocationLink(
                        mk_node("color:" + str(color)),
                        mk_node(
                            "viewLocation:"
                            + str(
                                (total_x // len(locations), total_y // len(locations))
                            )
                        ),
                    )
                )
            return observation

        elif label == "compassAngle":
            lca = self.last_compass_angle
            current = float(args[0])
            observation = []

            if not lca is None:
                if abs(0 - current) < abs(0 - lca):
                    observation = [mk_evaluation("compassAngleCloser")]

            self.last_compass_angle = current
            return observation

        return super().transform_percept(label, *args)

    def parse_world_state(
        self, ospace: Space, obs, reward: float, done: bool
    ) -> Tuple[List[Atom], Atom, bool]:
        reward = 1 if reward > 0.0 else -1

        return super().parse_world_state(ospace, obs, reward, done)
