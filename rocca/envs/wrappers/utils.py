from numbers import Number
from typing import Any

import numpy as np
from fastcore.basics import listify
from opencog.atomspace import Atom, is_a
from opencog.atomspace import types as AT
from opencog.type_constructors import *


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
