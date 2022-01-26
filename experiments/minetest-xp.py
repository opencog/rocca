#!/usr/bin/env python3

# Experiment to communicate with Minetest via Miney.
#
# Requirements:
#
# - Minetest, version 5.5.0
# - Miney https://miney.readthedocs.io/en/latest/index.html
# - luacmd https://github.com/prestidigitator/minetest-mod-luacmd
# - inspect https://github.com/kikito/inspect.lua
#
# Before launching minetest you might need to add
#
#      secure.trusted_mods = mineysocket, luacmd
#
# as well as
#
#      secure.enable_security = false
#
#    in your minetest.conf
#
# Usage:
#
# 1. Run minetest
# 2. Create game with mineysocket and optionally luacmd enabled
# 3. Run that script

import copy
import time
import miney

# Delay between each test (in second)
test_delay = 1.0


def log_and_wait(msg: str):
    print("============ {} ============".format(msg))
    time.sleep(test_delay)


####################
# Miney API Basics #
####################

# Connect Minetest
log_and_wait("Connect to minetest")
mt = miney.Minetest()
print("mt = {}".format(mt))

# We set the time to mid-day
log_and_wait("Set time to mid-day")
print("mt.time_of_day = {}".format(mt.time_of_day))
mt.time_of_day = 0.5
print("mt.time_of_day = {}".format(mt.time_of_day))

# Get Lua interpreter
log_and_wait("Get lua interpreter")
lua = miney.Lua(mt)
print("lua = {}".format(lua))

##########################
# Lua Interpreter Basics #
##########################

# Convert Python data into lua string.
#
# List of supported Python constructs (so far):
# - list
# - dict
#
# List of unsupported Python constructs (so far):
# - set
# - tuple
log_and_wait("Convert python data to lua")
data_list = ["123", "abc"]
data_list_dumps = lua.dumps(data_list)
print("python list = {}".format(data_list))
print("lua list dumps = {}".format(data_list_dumps))
data_dict = {"123": 123, "abc": 234}
data_dict_dumps = lua.dumps(data_dict)
print("python dict = {}".format(data_dict))
print("lua dict dumps = {}".format(data_dict_dumps))

# Helpers to print and run lua code
def lua_prt(code: str):
    print('Run lua code:\n"""\n{}\n"""'.format(code.strip("\n")))


def lua_prt_run(code: str):
    lua_prt(code)
    return lua.run(code)


# Run simple lua code
log_and_wait("Invoke minetest lua interpreter to calculate 1+2")
simple_lua_result = lua_prt_run("return 1 + 2")
print("simple_lua_result = {}".format(simple_lua_result))

# Return nil (should get None in Python)
log_and_wait("Return nil")
nil_result = lua_prt_run("return nil")
print("nil_result = {}".format(nil_result))

# Persistance
log_and_wait("Check persistance of lua interpreter")
lua.run("alpha = 10")
alpha = lua_prt_run("return alpha")
print("alpha = {}".format(alpha))

# Check if luacmd amd miney are pointing to the same interpreter.  You must enter
#
# /lua beta = 20
#
# in minetest before testing this.  If it return 20, then it should
# mean that luacmd points to the same lua interpreter as miney.
#
# Actually it does not!!! What is that supposed to mean?
log_and_wait("Check luacmd and miney points to same lua interpreter")
beta = lua_prt_run("return beta")
print("beta = {}".format(beta))

# # Include library (better use request_insecure_environment below)
# log_and_wait("Include lua library")
# inspect = lua_prt_run("inspect = require 'inspect'")
# print("inspect = {}".format(inspect))

# Include library by requesting insecure environment
log_and_wait("Include and test lua library (requesting insecure environment)")
test_inspect = """
ie = minetest.request_insecure_environment()
inspect = ie.require("inspect")
return inspect({1, 2, 3, 4})
"""
inspect_result = lua_prt_run(test_inspect)
print("inspect_result = {}".format(inspect_result))

# Inspect nil
log_and_wait("Inspect nil")
inspect_nil_result = lua_prt_run("return inspect(nil)")
print("inspect_nil_result = {}".format(inspect_nil_result))

# More inspect combined with loop and string concatenation.
log_and_wait("Inspect combined with loop and string concatenation")
test_inspect_comb = """
local t = {1, 2, 3, 4}
local msg = ''
for i, o in pairs(t) do
   msg = msg .. inspect(i) .. ':' .. inspect(o) .. ';'
end
return msg
"""
inspect_comb_result = lua_prt_run(test_inspect_comb)
print("inspect_comb_result = {}".format(inspect_comb_result))

# Chat
log_and_wait("Send chat to minetest")
chat_result = lua_prt_run('return minetest.chat_send_all("Hello Minetest")')
print("chat_result = {}".format(chat_result))

# Get registered nodes.  Miney cannot serialize minetest player, thus
# we return its string representation.
log_and_wait("Get registered nodes")
registered_nodes = lua_prt_run("return inspect(minetest.registered_nodes)")
print("registered_nodes = {}".format(registered_nodes))

###################################
# Player Action/Perception Basics #
###################################

# Testing player
player_name = "singleplayer"

# Check that the player exists
log_and_wait("Check that player exists")
player_exists_result = lua_prt_run(
    'return minetest.player_exists("{}")'.format(player_name)
)
print("player_exists = {}".format(player_exists_result))

# Create lua runner helper to run player methods
def player_lua_prt_run(code: str, player: str = "singleplayer"):
    get_player_code = 'local player = minetest.get_player_by_name("{}")'.format(player)
    full_code = get_player_code + "\n" + code.strip("\n")
    return lua_prt_run(full_code)


# Get the main player.  Miney cannot serialize minetest player, thus
# we return its string representation.
log_and_wait("Get player")
player_result = player_lua_prt_run("return inspect(player)")
print("player_result = {}".format(player_result))

# Get the name of the player (given its name, hehe)
log_and_wait("Get player name")
player_get_name_result = player_lua_prt_run("return player:get_player_name()")
print("player_get_name_result = {}".format(player_get_name_result))

# Get player's properties.  Miney cannot serialize minetest inventory,
# thus we return its string representation.
log_and_wait("Get player properties")
player_properties_result = player_lua_prt_run("return inspect(player:get_properties())")
print("player_properties_result = {}".format(player_properties_result))

# Get player position
log_and_wait("Get player position")
player_pos = player_lua_prt_run("return player:get_pos()")
print("player_pos = {}".format(player_pos))

# Get player's surrounding objects (likely only itself).  Miney cannot
# serialize minetest objects, thus we return their string
# representation.
log_and_wait("Get surrounding objects")
surrounding_objects_result = player_lua_prt_run(
    "return inspect(minetest.get_objects_inside_radius(player:get_pos(), 1.0))"
)
print("surrounding_objects_result = {}".format(surrounding_objects_result))

# Get the block (called node in minetest) at player's position
log_and_wait("Get node at player's position")
player_node_result = player_lua_prt_run("return minetest.get_node(player:get_pos())")
print("player_node_result = {}".format(player_node_result))

# Get metadata of node at player's position.  TODO: seems to return
# the player object!!!
log_and_wait("Get node metadata at player's position")
player_meta_result = player_lua_prt_run(
    "return inspect(minetest.get_meta(player:get_pos()))"
)
print("player_meta_result = {}".format(player_meta_result))

# Get player's nearest node of type air.
log_and_wait("Get player's nearest node of type air")
player_near_air_node_result = player_lua_prt_run(
    "return minetest.find_node_near(player:get_pos(), 1.0, 'air')"
)
print("player_near_air_node_result = {}".format(player_near_air_node_result))

# Get player's nearest node of type dry dirt.
log_and_wait("Get player's nearest node of type dry dirt")
player_near_dry_dirt_node_result = player_lua_prt_run(
    "return minetest.find_node_near(player:get_pos(), 10.0, 'default:dry_dirt')"
)
print("player_near_dry_dirt_node_result = {}".format(player_near_dry_dirt_node_result))

# Get player's nearest node of type desert sand.
log_and_wait("Get player's nearest node of type desert sand")
player_near_desert_sand_node_result = player_lua_prt_run(
    "return minetest.find_node_near(player:get_pos(), 10.0, 'default:desert_sand')"
)
print(
    "player_near_desert_sand_node_result = {}".format(
        player_near_desert_sand_node_result
    )
)

# Get player's nearest node of type dry dirt or desert sand.
log_and_wait("Get player's nearest node of type desert sand")
ddds = "{'default:dry_dirt', 'default:desert_sand'}"
player_near_node_result = player_lua_prt_run(
    "return minetest.find_node_near(player:get_pos(), 10.0, {})".format(ddds)
)
print("player_near_node_result = {}".format(player_near_node_result))

# Get player's nearest nodes under air.
log_and_wait("Get player's surrounding nodes")
player_lower_pos = {k: v - 5.0 for k, v in player_pos.items()}
player_upper_pos = {k: v + 5.0 for k, v in player_pos.items()}
lp = lua.dumps(player_lower_pos)
up = lua.dumps(player_upper_pos)
player_surrounding_nodes_result = player_lua_prt_run(
    "return minetest.find_nodes_in_area_under_air({}, {}, {})".format(lp, up, ddds)
)
print("player_surrounding_nodes_result = {}".format(player_surrounding_nodes_result))

# Get node drops.  Returns list of itemstrings that are dropped by
# `node` when dug with the item `toolname` (not limited to tools)
log_and_wait("Get node drops")
node_drops_result = lua_prt_run("return minetest.get_node_drops({}, nil)".format(ddds))
print("node_drops_result = {}".format(node_drops_result))

# Get node associated to near node
log_and_wait("Get node associated to near node")
associated_node = lua_prt_run(
    "return minetest.get_node({})".format(lua.dumps(player_near_node_result))
)
print("associated_node = {}".format(associated_node))

# Dig
log_and_wait("Dig")
player_dig_result = player_lua_prt_run(
    "return minetest.node_dig({}, {}, player)".format(
        lua.dumps(player_near_node_result), lua.dumps(associated_node)
    )
)
print("player_dig_result = {}".format(player_dig_result))

# Get player's inventory.  Miney cannot serialize minetest inventory,
# thus we return its string representation.
log_and_wait("Get player's inventory")
player_inventory_result = player_lua_prt_run(
    "return inspect(player:get_inventory():get_lists())"
)
print("player_inventory_result = {}".format(player_inventory_result))

# Get player's main inventory.  Miney cannot serialize minetest
# inventory, thus we return its string representation.
log_and_wait("Get player's main inventory")
player_main_inventory_result = player_lua_prt_run(
    "return inspect(player:get_inventory():get_list('main'))"
)
print("player_main_inventory_result = {}".format(player_main_inventory_result))

# Get player's first itemstack of its main inventory.
stack_index = 1
log_and_wait("Get player's first itemstack of main inventory")
player_first_itemstack_main_inventory_result = player_lua_prt_run(
    "return player:get_inventory():get_stack('main', {}):to_table()".format(stack_index)
)
print(
    "player_first_itemstack_main_inventory_result = {}".format(
        player_first_itemstack_main_inventory_result
    )
)

# Get player's first itemstack count of its main inventory.
log_and_wait("Get player's first itemstack count of main inventory")
player_first_itemstack_count_main_inventory_result = player_lua_prt_run(
    "return player:get_inventory():get_stack('main', {}):get_count()".format(
        stack_index
    )
)
print(
    "player_first_itemstack_count_main_inventory_result = {}".format(
        player_first_itemstack_count_main_inventory_result
    )
)

# Another attempt at retrieving player's inventory
log_and_wait("Get inventory formspec")
player_inventory_formspec_result = player_lua_prt_run(
    "return player:get_inventory_formspec()"
)
print("player_inventory_formspec_result = {}".format(player_inventory_formspec_result))

# Drop item from player inventory
log_and_wait("Drop first item from inventory")
itemstack = lua.dumps(player_first_itemstack_main_inventory_result)
drop_item_lua_code = """
local pl_inv = player:get_inventory()
local pl_stack = pl_inv:get_stack('main', {0})
local pl_stack_count = pl_stack:get_count()
local pl_leftover_stack = minetest.item_drop(pl_stack, player, {1})
pl_inv:set_stack('main', {0}, pl_leftover_stack)
return pl_leftover_stack:to_table()
-- TODO: minetest.item_drop actually drops the entire stack.
-- Below is a (failed to far) attempt to only drop one item of the stack.
--
-- local is_pl_stack_cleared = pl_stack:set_count(pl_stack_count - 1)
-- pl_inv:set_stack('main', {1}, pl_stack)
-- pl_inv:set_stack('main', {1}, pl_stack)
-- return pl_stack:to_table()
""".format(
    stack_index, lua.dumps(player_pos)
)
leftover_itemstack = player_lua_prt_run(drop_item_lua_code)
print("leftover_itemstack = {}".format(leftover_itemstack))

# TODO: minetest.item_eat, maybe

# Move abruptly player to a position.  Setting the continuous argument
# of move_to to true does not work for players (as explained in
# https://github.com/minetest/minetest/blob/master/doc/lua_api.txt).
log_and_wait("Move abruptly player to new position")
player_shifted_pos = copy.copy(player_pos)
player_shifted_pos["x"] += 1
print("player_shifted_pos = {}".format(player_shifted_pos))
move_player_to = "return player:move_to({})".format(lua.dumps(player_shifted_pos))
move_player_to_result = player_lua_prt_run(move_player_to)
print("move_player_to_result = {}".format(move_player_to_result))

# Come back smoothly to its original position.  TODO: use player_api
# to simulate walking.
log_and_wait("Move smoothly player to back to previous position")
player_move_back_result = player_lua_prt_run(
    "return player:add_velocity({x=-6.5, y=0, z=0})"
)
print("player_move_back_result = {}".format(player_move_back_result))

# Go slowly to the previous position
log_and_wait("Move smoothly player to back to previous position")
player_move_back_result = player_lua_prt_run(
    "return player:add_velocity({x=3.0, y=0, z=0})"
)
print("player_move_back_result = {}".format(player_move_back_result))

# Go very fast in the same direction
log_and_wait("Move smoothly player to back to previous position")
player_move_back_result = player_lua_prt_run(
    "return player:add_velocity({x=20.0, y=0, z=0})"
)
print("player_move_back_result = {}".format(player_move_back_result))

# Go to another direction
log_and_wait("Move smoothly player to back to previous position")
player_move_back_result = player_lua_prt_run(
    "return player:add_velocity({x=0, y=0, z=6.5})"
)
print("player_move_back_result = {}".format(player_move_back_result))

# Get player horizontal look angle
log_and_wait("Get player look horizontal angle")
player_look_horizontal_angle = player_lua_prt_run("return player:get_look_horizontal()")
print("player_look_horizontal_angle = {}".format(player_look_horizontal_angle))

# Set player horizontal look angle (turn, though not smoothly)
log_and_wait("Set player look horizontal angle (+0.1)")
new_player_look_horizontal_angle = player_look_horizontal_angle + 0.1
set_player_look_horizontal_result = player_lua_prt_run(
    "return player:set_look_horizontal({})".format(new_player_look_horizontal_angle)
)
print(
    "set_player_look_horizontal_result = {}".format(set_player_look_horizontal_result)
)

# TODO
# * `minetest.place_node(pos, node)`
#     * Place node with the same effects that a player would cause
# * `minetest.punch_node(pos)`
#     * Punch node with the same effects that a player would cause

# Jump
log_and_wait("Jump")
player_jump_result = player_lua_prt_run("return player:add_velocity({x=0, y=6.5, z=0})")
print("player_jump_result = {}".format(player_jump_result))

# Jump higher
log_and_wait("Jump higher")
player_jump_result = player_lua_prt_run("return player:add_velocity({x=0, y=13, z=0})")
print("player_jump_result = {}".format(player_jump_result))

# Player control
log_and_wait("Get player control")
player_control_result = player_lua_prt_run("return player:get_player_control()")
print("player_control_result = {}".format(player_control_result))
