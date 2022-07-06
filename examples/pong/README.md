# Pong

## Description

Agent for Pong-v0 gym env.

## Status

Unmantained.  Would be nice to maintain it though, it is a simple yet
already challenging environment due to the delay between actions and
rewards.  I.e. the agent needs to move the paddle early on in order to
avoid missing the ball later on.

## Requirements

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

## Usage

```bash
python pong.py
```
