# Examples

Contains examples of OpencogAgent running in various environments.

## Chase

### Description

Simplistic home-brewed gym env.  2 squares, with one pellet of food.
The agent must chase and eat the food pellet to accumulate rewards.

### Status

The agent is able to learn mono and poly action plans to chase and eat
the food pellet, either via purely direct evaluations (pattern
mining), or a combination of direct and indirect evaluations by
building more complex plans out of simpler plans using temporal
deduction.  The optimal strategy is achieved with the following 2
plans:

1. `PelletPosition(None) ∧ AgentPosition(LeftSquare) ∧ do(GoRight) ⩘ do(Eat) ↝ Reward(1)`
2. `PelletPosition(None) ∧ AgentPosition(RightSquare) ∧ do(GoLeft) ⩘ do(Eat) ↝ Reward(1)`

represented in Atomese as

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00669975)
  (VariableSet
  ) ; [80075fffffff523a][3]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][3]
  ) ; [da5f815ba9d4009f][3]
  (BackSequentialAndLink
    (SLink
      (ZLink
      ) ; [800fbffffffe8ce4][3]
    ) ; [da5f815ba9d4009f][3]
    (AndLink (stv 0.03 0.2)
      (EvaluationLink (stv 0.12 0.2)
        (PredicateNode "Pellet Position") ; [56e6ab0f525cb504][3]
        (ConceptNode "None") ; [68c616828fa0f8e6][3]
      ) ; [a609b2dbedb7904e][3]
      (ExecutionLink
        (SchemaNode "Go Right") ; [51c7a48fd94d12d8][3]
      ) ; [c29bf0559d1ad8ec][3]
      (EvaluationLink (stv 0.37 0.2)
        (PredicateNode "Agent Position") ; [3fdca752fd5e5335][3]
        (ConceptNode "Left Square") ; [586f8a0db3b1388a][3]
      ) ; [ec8ec4a729ccab8c][3]
    ) ; [f07c6a1c8318095c][3]
    (ExecutionLink
      (SchemaNode "Eat") ; [3fe4e22345c3679f][3]
    ) ; [9efce1dc8918c209][3]
  ) ; [b7ee6db6cd3a43bf][3]
  (EvaluationLink (stv 0.12 0.2)
    (PredicateNode "Reward") ; [155bb4d713db0d51][3]
    (NumberNode "1") ; [2cf0956d543cff8e][3]
  ) ; [d3cee8bdda06ffcb][3]
) ; [cac8b7b130c2115c][3]
```

```scheme
(BackPredictiveImplicationScopeLink (stv 1 0.00447761)
  (VariableSet
  ) ; [80075fffffff523a][3]
  (SLink
    (ZLink
    ) ; [800fbffffffe8ce4][3]
  ) ; [da5f815ba9d4009f][3]
  (BackSequentialAndLink
    (SLink
      (ZLink
      ) ; [800fbffffffe8ce4][3]
    ) ; [da5f815ba9d4009f][3]
    (AndLink (stv 0.02 0.2)
      (EvaluationLink (stv 0.12 0.2)
        (PredicateNode "Pellet Position") ; [56e6ab0f525cb504][3]
        (ConceptNode "None") ; [68c616828fa0f8e6][3]
      ) ; [a609b2dbedb7904e][3]
      (ExecutionLink
        (SchemaNode "Go Left") ; [7ca250f2efc2e872][3]
      ) ; [c7fb76d9605d5db5][3]
      (EvaluationLink (stv 0.63 0.2)
        (PredicateNode "Agent Position") ; [3fdca752fd5e5335][3]
        (ConceptNode "Right Square") ; [6dd382acb6aa376e][3]
      ) ; [c9fcc2094e0150df][3]
    ) ; [e1b8c69b9a37aa32][3]
    (ExecutionLink
      (SchemaNode "Eat") ; [3fe4e22345c3679f][3]
    ) ; [9efce1dc8918c209][3]
  ) ; [b470646c0c97b387][3]
  (EvaluationLink (stv 0.12 0.2)
    (PredicateNode "Reward") ; [155bb4d713db0d51][3]
    (NumberNode "1") ; [2cf0956d543cff8e][3]
  ) ; [d3cee8bdda06ffcb][3]
) ; [8f007faa792823de][3]
```

However, due to balancing exploration and exploitation, since ROCCA
also learns many more sub-optimal plans and has not full confidence
regarding the optimal plans (full confidence can only be
experientially obtained with infinitely many observations), it does
not achieve perfect score (which is 100).  It however achieves greater
and greater scores after each learning session due to the confidences
of the optimal plans growing faster than that of the others.

### Usage

```bash
python chase.py
```

## CartPole

### Description

CartPole-v1 gym env.

### Status

No learning for now, the cognitive schemantics are hardcoded to move
the cart left or right when the pole deviates from the origin.

### Usage

```bash
python cartpole.py
```

## Pong

### Description

Pong-v0 gym env.

### Status

Unmantained.  Would be nice to maintain it though, it is a simple yet
already challenging environment the delay between actions and rewards.
I.e. the agent needs to move the paddle early on in order to avoid
missing the ball later on.

### Requirements

You need to install the `atari` subpackage of gym, which you can do with

```
pip install gym[atari]
```

Additionally you need to install the Atari ROMS, download a ROM pack
from

http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html

unrar it under the current folder

```
unrar e Roms.rar
```

It should create 2 zip files.  Then you can install the ROMS with the
following command

```
python -m atari_py.import_roms <ROM_FOLDER>
```

You'll need to have `atari-py` properly installed for that.

### Usage

```bash
python pong.py
```

## Chase Malmo

### Description

Port of chase for Malmo.  Food pellets are replaced by diamonds.

### Status

Unmaintained.  As a close alternative, see the jupyter notebook file
`02_minerl_navigate_agent.ipynb` under the root folder which relies on
`minerl`.

### Usage

1. Compile malmo https://github.com/Microsoft/malmo, see
   https://github.com/microsoft/malmo/blob/master/doc/build_linux.md

2. After compiling, copy
   `<MALMO>/build/install/Python_Examples/MalmoPython.so` and
   `<MALMO>/build/install/Python_Examples/malmoutils.py` under
   `<ROCCA>/examples/malmo`.

3. Launch `<MALMO>/Minecraft/launchClient.sh`.

4. Launch chase malmo `python chase_malmo.py`.
