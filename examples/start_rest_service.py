from opencog.web.api.apimain import RESTAPI
from opencog.atomspace import AtomSpace
from opencog.type_constructors import set_default_atomspace
from opencog.scheme import scheme_eval

from requests import post


def start_restapi(PORT="5000", IP_ADDRESS="127.0.0.1"):
    atomspace = AtomSpace()
    set_default_atomspace(atomspace)
    load_modules(atomspace)
    api = RESTAPI(atomspace)
    api.run(host=IP_ADDRESS, port=PORT)


def load_modules(atomspace):
    scheme_eval(atomspace, "(use-modules (opencog))")
    scheme_eval(atomspace, "(use-modules (opencog exec))")
    scheme_eval(atomspace, "(use-modules (opencog pln))")
    scheme_eval(atomspace, "(use-modules (opencog spacetime))")
    scheme_eval(atomspace, "(use-modules (opencog persist))")


if __name__ == "__main__":
    start_restapi()
