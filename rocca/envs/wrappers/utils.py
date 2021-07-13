from numbers import Number
from typing import Any

import numpy as np
from fastcore.basics import listify
from opencog.atomspace import Atom, is_a
from opencog.atomspace import types as AT
from opencog.type_constructors import *
from opencog.spacetime import *


def mk_action(name, value):
    if isinstance(value, np.ndarray):
        value = listify(value.tolist())
        return ExecutionLink(SchemaNode(name), mk_list(*value))

    return ExecutionLink(SchemaNode(name), mk_node(value))


def mk_node(name):
    if isinstance(name, Number):
        return NumberNode(str(name))
    if isinstance(name, str):
        return ConceptNode(name)
    raise RuntimeError("Error: Unknown type.")

last_compassAngle = None
def convert_percept(predicate, *args):
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
        #print (args, type(args))
        #args = ["some image"]
        from collections import defaultdict
        colors = defaultdict(list)
        
        for y in range(0, 64):
            for x in range(0, 64):
                color = args[y][x]
                rounded_color = tuple([subpixel // 25 * 25 for subpixel in color])
                colors[rounded_color].append((x, y))
        
        #print(f"{len(colors.keys())} colors in this frame")
        #args = ["some image"]
        links = []
        for (color, locations) in colors.items():
            total_x = total_y = 0
            for (x, y) in locations:
                total_x += x
                total_y += y
            links.append(AtLocationLink(mk_node("color:"+str(color)), mk_node("viewLocation:"+str((total_x//len(locations), total_y//len(locations))))))
        #print(links)
        return links
    
    elif predicate == "Reward":
        if float(args[0]) > 0:
            args = [1]
        elif float(args[0]) < 0:
            args = [-1]
            
    elif predicate == "compassAngle":
        global last_compassAngle
        lca = last_compassAngle
        current = float(args[0])
        links = []
        
        if not lca is None:
            if abs(0 - current) < abs(0 - lca):
                links = [mk_evaluation("compassAngleCloser")]
                print(links)
        
        last_compassAngle = current
        return links

    return [mk_evaluation(predicate, *args)]

def mk_evaluation(predicate, *args):
    pred = PredicateNode(predicate)
    if len(args) == 1:
        if not isinstance(args[-1], bool):
            return EvaluationLink(pred, mk_node(args[-1]))
        else:
            return EvaluationLink(pred, mk_node("agent"))

    arg_listlink = mk_list(*args)
    return EvaluationLink(pred, arg_listlink)


def mk_list(*args) -> ListLink:
    processed_items = []
    for arg in args:
        if isinstance(arg, list):
            processed_items.append(mk_list(*arg))
        else:
            processed_items.append(mk_node(arg))

    return ListLink(*processed_items)


def mk_minerl_single_action(env, name: str, value: Any):
    """Fill in information about one action into a no-op"""

    noop = env.action_space.noop()
    actions = [mk_action(k, noop[k]) for k in noop if k != name]
    actions.append(mk_action(name, value))

    return actions


def minerl_single_action(env, action):
    """Insert a single action into a no-op

        env: Gym environment or a wrapped gym environment.
        action: action of the form `Execution (Schema name) args`
    """
    noop = env.action_space.noop()
    actions = [mk_action(k, noop[k]) for k in noop if k != action.out[0].name]
    actions.append(action)

    return actions

def to_python(atom: Atom):
    """Return a Pythonic data representation of an atom"""
    if is_a(atom.type, AT.ListLink):
        # HACK: here I just turn lists into numpy arrays because MineRL expects that
        return np.array([to_python(a) for a in atom.out])
    if is_a(atom.type, AT.NumberNode):
        return float(atom.name)
    return atom.name
