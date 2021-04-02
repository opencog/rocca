OpenCog wrapper around OpenAI Gym.

Highly experimental at this stage.

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
- OpenAI Gym https://gym.openai.com/
- Malmo https://github.com/Microsoft/malmo (after compiling Malmo, see
  https://github.com/microsoft/malmo/blob/master/doc/build_linux.md,
  copy `<MALMO>/build/install/Python_Examples/MalmoPython.so` and
  `<MALMO>/build/install/Python_Examples/malmoutils.py` under
  `<ROCCA>/examples/malmo`, then launch
  `<MALMO>/Minecraft/launchClient.sh` before running the malmo
  example).

## Install

In the root folder enter the following command

```bash
pip install .
```

## Usage

For now a gym agent defined under the `agent` folder is provided that
can used to implement agents for given environments.  See the examples
under the `examples` folder.

