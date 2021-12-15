from numbers import Number
from typing import *

import numpy as np

from fastcore.basics import listify
from gym import Env

from opencog.atomspace import Atom, is_a, TruthValue
from opencog.atomspace import types as AT
from opencog.type_constructors import *


def mk_action(name: str, value) -> Atom:  # with Python 3.10 value: np.ndarray | Any
    if isinstance(value, np.ndarray):
        value = listify(value.tolist())
        return ExecutionLink(SchemaNode(name), mk_list(*value))

    return ExecutionLink(SchemaNode(name), mk_node(value))


def mk_node(name: str) -> Atom:
    if isinstance(name, Number):
        return NumberNode(str(name))
    if isinstance(name, str):
        return ConceptNode(name)
    raise RuntimeError("Error: Unknown type.")


def mk_evaluation(predicate_name: str, *args) -> Atom:
    pred = PredicateNode(predicate_name)
    if len(args) == 1:
        if not isinstance(args[-1], bool):
            return EvaluationLink(pred, mk_node(args[-1]))
        else:
            tv = TruthValue(1.0, 1.0) if args[-1] else TruthValue(0.0, 1.0)
            return EvaluationLink(pred, mk_node("agent"), tv=tv)

    arg_listlink = mk_list(*args)
    return EvaluationLink(pred, arg_listlink)


def mk_list(*args) -> Atom:
    processed_items = []
    for arg in args:
        if isinstance(arg, list):
            processed_items.append(mk_list(*arg))
        else:
            processed_items.append(mk_node(arg))

    return ListLink(*processed_items)


def mk_minerl_single_action(env: Env, name: str, value: Any) -> list[Atom]:
    """Fill in information about one action into a no-op"""

    noop = env.action_space.noop()
    actions = [mk_action(k, noop[k]) for k in noop if k != name]
    actions.append(mk_action(name, value))

    return actions


def to_python(atom: Atom):  # with Python 3.10 -> np.ndarray | float | Any
    """Return a Pythonic data representation of an atom"""

    if is_a(atom.type, AT.ListLink):
        # HACK: here I just turn lists into numpy arrays because MineRL expects that
        return np.array([to_python(a) for a in atom.out])
    if is_a(atom.type, AT.NumberNode):
        return float(atom.name)
    return atom.name
