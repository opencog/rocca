from numbers import Number

from opencog.type_constructors import *


def mk_action(name, value):
    return ExecutionLink(SchemaNode(name),
                         NumberNode(str(value)))


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
    arg_list = [mk_node(arg) for arg in args]
    return EvaluationLink(pred, ListLink(*arg_list))
