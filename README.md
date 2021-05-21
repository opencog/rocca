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
