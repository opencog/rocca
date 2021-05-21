# Examples

Contains examples of OpencogAgent running in various environments.

## Chase

### Description

Simplistic home-brewed gym env.  2 squares, with one pellet of food.
The agent must chase and eat the food pellet to accumulate rewards.

### Status

The agent is able to learn mono and poly action plans to chase and eat
the food pellet.  However it is so far only able to learn them by
direct observations of action sequences and will not be able to reason
about action sequences it has never observed.  That is coming next.

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
