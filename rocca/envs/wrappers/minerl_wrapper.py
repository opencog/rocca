from .utils import *
from .gym_wrapper import GymWrapper


class MineRLWrapper(GymWrapper):
    def __init__(self, env, action_list=[]):
        super().__init__(env, action_list)
        self.last_compassAngle = None

    def convert_percept(self, predicate, *args):
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
        if predicate == "pov":
            # print (args, type(args))
            # args = ["some image"]
            from collections import defaultdict

            colors = defaultdict(list)

            for y in range(0, 64):
                for x in range(0, 64):
                    color = args[y][x]
                    rounded_color = tuple([subpixel // 25 * 25 for subpixel in color])
                    colors[rounded_color].append((x, y))

            # print(f"{len(colors.keys())} colors in this frame")
            # args = ["some image"]
            links = []
            for (color, locations) in colors.items():
                total_x = total_y = 0
                for (x, y) in locations:
                    total_x += x
                    total_y += y
                links.append(
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
            # print(links)
            return links

        elif predicate == "Reward":
            if float(args[0]) > 0:
                args = [1]
            elif float(args[0]) < 0:
                args = [-1]

        elif predicate == "compassAngle":
            lca = self.last_compassAngle
            current = float(args[0])
            links = []

            if not lca is None:
                if abs(0 - current) < abs(0 - lca):
                    links = [mk_evaluation("compassAngleCloser")]

            self.last_compassAngle = current
            return links

        return [mk_evaluation(predicate, *args)]
