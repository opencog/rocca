# Rational OpenCog Controlled Agent

## Description

Rational OpenCog Controlled Agent, or ROCCA, is a project aiming at
creating an opencog agent that acts rationally in OpenAI Gym
environments (including Minecraft via minerl).

At its core it relies on PLN (Probabilistic Logic Networks) for both
learning and planning.  In practice most of the learning is however
handled by the pattern miner, which can be seen as a specialized form
of PLN reasoning.  Planning, the discovery of cognitive schematics, is
handled by PLN and its temporal reasoning rule base.  Decision is
currently a hardwired module, heavily inspired by OpenPsi with a more
rational sampling procedure (Thompson Sampling for better exploitation
vs exploration tradeoff).

## Status

For now learning is only able to abstract temporal patterns based on
directly observable events.  That is the agent is able to notice that
particular action sequences in certain contexts tend to be followed by
rewards, however it is not, as of right now, able to reason about
action sequences that it has never observed.  This requires Temporal
Deduction, currently under development.

Once Temporal Deduction is complete we still have a lot of things to
add such as

1. More sophisticated temporal and then spatial inference rules.
2. ECAN, for Attention Allocation, to dynamically restrict the
   atomspace to subsets of items to process/pay-attention-to.
3. Record attention spreading to learn/improve Hebbian links.
4. Concept creation and schematization (crystallized attention
   allocation).
5. Record internal processes, not just attention spreading, as
   percepta to enable deeper forms of instrospective reasoning.
6. Plan internal actions, not just external, to enable self-growth.

## Requirements

OpenCog tools

- cogutil
- atomspace
- ure
- spacetime
- pln
- miner
- [optional] cogserver
- [optional] attention
- [optional] opencog

Third party tools

- Python 3
- python-orderedmultidict https://pypi.org/project/orderedmultidict/
- fastcore https://fastcore.fast.ai
- OpenAI Gym https://gym.openai.com/
- MineRL https://minerl.io
- nbdev https://nbdev.fast.ai

## Install

In the root folder enter the following command:

```bash
pip install -e .
```

## How to use

A gym agent defined under the `rocca/agents` folder is provided that
can used to implement agents for given environments.  See the examples
under the `examples` folder.

There are Jupyter notebooks provided for experimentation as well.

## Develop

### nbdev workflow

You do not have to use `nbdev` to work with the code under the `rocca` directory.
You should use it though if you want to work with Jupyter notebooks, the repository is setup to use certain
utilities to clean them from unnecessary metadata when committing.

One important mention is that `README.md` is now generated from `index.ipynb` by `nbdev_build_docs` command.
Thus, you should not edit `README.md` directly.

Exports from notebooks are generated with `nbdev_build_lib`. Changes to the exported code can be synchronized
back to notebooks with `nbdev_update_lib`.

### VS Code devcontainer

This repository contains configuration that can be automatically used by VS Code to build a Docker container
that will have the contents of this repository mounted and all dependencies installed. VS Code should ask you
to reopen the directory in a container if you have its _Remote-Containers_ extension installed.

Running the provided configuration will start a JupyterLab instance that will be available on port 8888.

**You have to inspect `.devcontainer/docker-compose.yml` and take care to set environment variables mentioned there.**

There is also the `.devcontainer/docker-compose-custom.yml` that you can use to add your own configuration, matching your
personal needs.
