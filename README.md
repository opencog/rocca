# Rational OpenCog Controlled Agent

<p align="left">
   <a href="https://github.com/opencog/rocca/actions">
      <img alt="CI Status" src="https://github.com/opencog/rocca/actions/workflows/main.yml/badge.svg">
   </a>
   <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
   </a>
</p>

## Description

Rational OpenCog Controlled Agent, or ROCCA, is a project aiming at
creating an opencog agent that acts rationally in OpenAI Gym
environments (including Minecraft via MineRL and Malmo).

At its core it relies on PLN (Probabilistic Logic Networks) for both
learning and planning.  In practice most of the learning is however
handled by the pattern miner, which can be seen as a specialized form
of PLN reasoning.  Planning, the discovery of cognitive schematics, is
handled by PLN and its temporal reasoning rule base.  Decision is
currently a hardwired module, heavily inspired by OpenPsi with a more
rational sampling procedure (Thompson Sampling for better exploitation
vs exploration tradeoff).

## Status

For now learning is able to

1. Discover temporal patterns based on directly observable events via
   the pattern miner.
2. Turn these temporal patterns into plans (cognitive schematics).
3. Combine these plans to form new plans, possibly composed of new
   action sequences, via temporal deduction.

The next steps are

1. Add more sophisticated temporal (including dealing with longs lags
   between cause and effect) and then spatial inference rules.
2. Integrate ECAN, for Attention Allocation, to dynamically restrict
   the atomspace to subsets of items to process/pay-attention-to.
3. Record attention spreading to learn/improve Hebbian links.
4. Carry concept creation and schematization (crystallized attention
   allocation).
5. Record internal processes, not just attention spreading, as
   percepta to enable deeper forms of instrospective reasoning.
6. Plan internal actions, not just external, to enable self-growth.

## Requirements

OpenCog tools

- cogutil (tested with revision 555a003)
- atomspace (tested with revision 396e1e7)
- unify (tested with revision 1e93141)
- ure (tested with revision 4e01b02)
- spacetime (tested with revision 962862c)
- pln (tested with revision 08c100f)
- miner (tested with revision 15befc4)

Third party tools

- Python 3.10 (or Python 3.8 see below)
- jupyter notebook
- python-orderedmultidict https://pypi.org/project/orderedmultidict/
- fastcore https://fastcore.fast.ai
- OpenAI Gym https://gym.openai.com/
- MineRL https://minerl.io
- nbdev https://nbdev.fast.ai
- black https://pypi.org/project/black/

### Python 3.10 vs 3.8

Python 3.10 offers a better out-of-the-box type annotation system than
Python 3.8 and is thus the default required version.  However you may
still use Python 3.8 by checking out the
[python-3.8-compatible](https://github.com/opencog/rocca/tree/python-3.8-compatible)
branch.  Beware that such Python 3.8 branch may not be as well
maintained as the master.

## Install

In the root folder enter the following command (you might need to be
root depending on your system):

```bash
pip install -e .
```

For the tools used for development:
```bash
pip install -r requirements-dev.txt
```

## How to use

An OpencogAgent defined under the `rocca/agents` folder is provided
that can used to implement agents for given environments.  See the
examples under the `examples` folder.

There are Jupyter notebooks provided for experimentation as well.  To
run them call jupyter notebook on a ipynb file, such as

```bash
jupyter notebook 01_cartpole.ipynb
```

### TensorBoard support

Some experiments, notably the notebooks, use TensorBoard via the
`tensorboardX` library to store event files that show certain metrics
over time for training / testing (for now it's just rewards).

By default, event files will be created under the
`runs/<datetime><comment>` directory. You can invoke `tensorboard
--logdir runs` from the project root to start an instance that will
see all the files under that directory. Open your browser to
`http://localhost:6006` to see its interface.

## Develop

If you write code in notebooks that is exported (has the `#export`
comment on top of the cell), remember to invoke `nbdev_build_lib` to
update the library. Remember to use `black` for formatting, you can
invoke `black .` from the project root to format everything.

You can also use the Makefile for your convenience, invoking `make
rocca` will do both of the above in sequence.

### Development container

The `.devcontainer` folder has configuration for [VS Code
devcontainer](https://code.visualstudio.com/docs/remote/containers)
functionality. You can use it to setup a development environment very
quickly and regardless of the OS you use.

- The container has a JupyterLab instance running on the port 8888.
- The container has a VNC server running on the port 5901.
- The password for the VNC server started in the container is
  `vncpassword`. You can use any VNC client to see the results of
  rendering Gym environments this way.

## Tests

### Static type checking

Using [type annotations](https://mypy.readthedocs.io/en/stable/getting_started.html)
is highly encouraged.  One can type check the entire Python ROCCA code
by calling

```
tests/mypy.sh
```

from the root folder.

To only type check some subfolder, you may call `mypy.sh` from that
subfolder.  For instance to type check the `examples` subfolder

```
cd examples
../tests/mypy.sh
```

Or directly run `mypy` on files or directories, such as

```
mypy rocca/rocca/agents/core.py
```

`tests/mypy.sh` merely calls mypy on all python files in the directory
from which it is called while filtering out some error messages.

### Unit tests

Simply run `pytest` in the root folder.

## References

There is no ROCCA paper per se yet.  In the meantime here is a list of related references

+ [An Inferential Approach to Mining Surprising Patterns in Hypergraphs, Nil Geisweiller et Ben Goertzel](https://www.researchgate.net/publication/334769428_An_Inferential_Approach_to_Mining_Surprising_Patterns_in_Hypergraphs)
+ [Partial Operator Induction with Beta Distributions, Nil Geisweiller](https://raw.githubusercontent.com/ngeiswei/papers/master/PartialBetaOperatorInduction/PartialBetaOperatorInduction.pdf)
+ [Thompson Sampling is Asymptotically Optimal in General Environments, Jan Leike et al](http://auai.org/uai2016/proceedings/papers/20.pdf)
+ [Draft about temporal reasoning in PLN, Nil Geisweiller](https://github.com/opencog/pln/blob/master/opencog/pln/rules/temporal/temporal-reasoning.md)
+ [Presentation and Demo of Temporal and Procedural Reasoning with OpenCog, AGI-21, Nil Geisweiller et Hedra Yusuf](https://odysee.com/@ngeiswei:d/AGI-21-Temporal-Procedural-Reasoning-Nil-Geisweiller-Hedra-Yusuf:6)
+ [References about OpenCog including PLN](https://wiki.opencog.org/w/Background_Publications)
