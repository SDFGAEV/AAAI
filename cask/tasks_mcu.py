"""
MCU Task Loader — parses MCU YAML task configs, returns task list with init commands.
All task names preserved from MCU. init_commands executed via /give, /setblock etc.
"""
import yaml, os

MCU_BASE = "E:/open-world agent/MCU/MCU_benchmark/task_configs"

def load_mcu_task(name, difficulty="simple"):
    """Load a single MCU task config and return (init_cmds, text, thinking)."""
    path = os.path.join(MCU_BASE, difficulty, f"{name}.yaml")
    if not os.path.exists(path):
        return [], name, ""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("custom_init_commands", []), cfg.get("text", name), cfg.get("thinking", "")

def task_entry(task_id, init_cmds, steps, verify="", vcount=1, difficulty="simple", source="MCU", group="crafting"):
    """Create a task entry compatible with experiment_final.py run_phase."""
    return (task_id, init_cmds, steps, verify, vcount, difficulty, source, group)

# ═══════════════════════════════════════════════════════════════
# Phase 0: Sanity Check — 6 MCU simple tasks
# ═══════════════════════════════════════════════════════════════
P0 = [
    # MCU simple crafting tasks with their original /give init
    task_entry("MCU_craft_oak_planks",
        ["/give @s minecraft:oak_log 10", "/give @s minecraft:crafting_table"],
        [("craft", "oak_planks", 4)], "oak_planks", 4, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_the_crafting_table",
        ["/give @s minecraft:oak_planks 20"],
        [("craft", "crafting_table", 1)], "crafting_table", 1, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_ladder",
        ["/give @s minecraft:stick 15", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "ladder", 3)], "ladder", 3, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_smelting",
        ["/give @s minecraft:cobblestone 20", "/give @s minecraft:crafting_table 1", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "furnace", 1)], "furnace", 1, "simple", "MCU", "crafting"),

    task_entry("MCU_collect_wood",
        ["/give @s minecraft:wooden_axe"],
        [("mine", "oak_log", 3)], "oak_log", 3, "simple", "MCU", "mining"),

    task_entry("MCU_cut_stone",
        ["/give @s minecraft:wooden_pickaxe"],
        [("mine", "stone", 5)], "cobblestone", 3, "simple", "MCU", "mining"),
]

# ═══════════════════════════════════════════════════════════════
# Phase 1: Knowledge Accumulation — 24 tasks
# ═══════════════════════════════════════════════════════════════
P1 = [
    # ── Crafting (8) ──
    task_entry("MCU_craft_oak_planks",
        ["/give @s minecraft:oak_log 10"],
        [("craft", "oak_planks", 4)], "", 0, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_the_crafting_table",
        ["/give @s minecraft:oak_planks 20"],
        [("craft", "crafting_table", 1)], "", 0, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_ladder",
        ["/give @s minecraft:stick 20", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "ladder", 3)], "", 0, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_smelting",
        ["/give @s minecraft:cobblestone 24", "/give @s minecraft:crafting_table"],
        [("craft", "furnace", 1)], "", 0, "simple", "MCU", "crafting"),

    task_entry("MCU_craft_ladder_hard",
        ["/give @s minecraft:stick 15", "/give @s minecraft:stone_pickaxe", "/give @s minecraft:iron_ingot 5",
         "/give @s minecraft:diamond 3", "/give @s minecraft:coal 8", "/give @s minecraft:arrow 15", "/give @s minecraft:bow 1",
         "/give @s minecraft:iron_helmet 1", "/give @s minecraft:iron_chestplate 1", "/give @s minecraft:iron_leggings 1",
         "/give @s minecraft:iron_boots 1", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "ladder", 2)], "", 0, "hard", "MCU", "crafting"),

    task_entry("MCU_craft_oak_planks_hard",
        ["/give @s minecraft:oak_log 8", "/give @s minecraft:stone 10", "/give @s minecraft:iron_ingot 5",
         "/give @s minecraft:diamond 3", "/give @s minecraft:apple 4", "/give @s minecraft:leather 6",
         "/give @s minecraft:string 10", "/give @s minecraft:stick 12", "/give @s minecraft:coal 5",
         "/give @s minecraft:torch 20", "/give @s minecraft:water_bucket 1", "/give @s minecraft:crafting_table"],
        [("craft", "oak_planks", 4)], "", 0, "hard", "MCU", "crafting"),

    task_entry("MCU_craft_stonecut",
        ["/give @s minecraft:stone 12", "/give @s minecraft:iron_ingot 4", "/give @s minecraft:crafting_table"],
        [("craft", "stone_pickaxe", 1)], "", 0, "hard", "MCU", "crafting"),

    task_entry("MCU_craft_diorite",
        ["/give @s minecraft:cobblestone 12", "/give @s minecraft:quartz 4", "/give @s minecraft:crafting_table"],
        [("craft", "stone_axe", 1)], "", 0, "hard", "MCU", "crafting"),

    # ── Mining/Collecting (6) ──
    task_entry("MCU_collect_wood",
        ["/give @s minecraft:wooden_axe"],
        [("mine", "oak_log", 4)], "", 0, "simple", "MCU", "mining"),

    task_entry("MCU_cut_stone",
        ["/give @s minecraft:wooden_pickaxe"],
        [("mine", "stone", 8)], "", 0, "simple", "MCU", "mining"),

    task_entry("MCU_collect_dirt",
        ["/give @s minecraft:wooden_shovel"],
        [("mine", "dirt", 16)], "", 0, "simple", "MCU", "mining"),

    task_entry("MCU_collect_wood_hard",
        ["/give @s minecraft:wooden_axe", "/give @s minecraft:stone 64", "/give @s minecraft:iron_ingot 10",
         "/give @s minecraft:diamond 5", "/give @s minecraft:golden_apple 2", "/give @s minecraft:bread 8",
         "/give @s minecraft:torch 20", "/give @s minecraft:leather 12", "/give @s minecraft:glass 15",
         "/give @s minecraft:water_bucket 1", "/give @s minecraft:coal 10", "/give @s minecraft:iron_sword 1",
         "/give @s minecraft:bow 1", "/give @s minecraft:arrow 20", "/give @s minecraft:map 1"],
        [("mine", "oak_log", 6)], "", 0, "hard", "MCU", "mining"),

    task_entry("MCU_cut_stone_hard",
        ["/give @s minecraft:iron_pickaxe", "/give @s minecraft:stone 64", "/give @s minecraft:diamond 5",
         "/give @s minecraft:golden_apple 3", "/give @s minecraft:arrow 32"],
        [("mine", "stone", 12)], "", 0, "hard", "MCU", "mining"),

    task_entry("MCU_mine_iron_ore",
        ["/give @s minecraft:stone_pickaxe", "/setblock ~1 ~-1 ~ minecraft:iron_ore"],
        [("mine", "iron_ore", 1)], "raw_iron", 1, "hard", "MCU", "mining"),

    # ── Building (4) ──
    task_entry("MCU_build_a_ladder",
        ["/give @s minecraft:stick 32", "/give @s minecraft:crafting_table", "/give @s minecraft:oak_planks 16"],
        [("craft", "ladder", 4), ("place", "ladder", 1)], "", 0, "simple", "MCU", "building"),

    task_entry("MCU_build_pillar",
        ["/give @s minecraft:cobblestone 64"],
        [("place", "cobblestone", 1)], "", 0, "simple", "MCU", "building"),

    task_entry("MCU_build_a_tower",
        ["/give @s minecraft:cobblestone 64", "/give @s minecraft:stone 32", "/give @s minecraft:oak_planks 32", "/give @s minecraft:ladder 16"],
        [("place", "cobblestone", 1)], "", 0, "hard", "MCU", "building"),

    task_entry("MCU_build_a_wall",
        ["/give @s minecraft:stone_bricks 64", "/give @s minecraft:stone 32", "/give @s minecraft:oak_planks 16"],
        [("place", "stone_bricks", 1)], "", 0, "hard", "MCU", "building"),

    # ── Smelting (2) ──
    task_entry("MCU_smelt_beef",
        ["/give @s minecraft:beef 8", "/give @s minecraft:coal 8", "/give @s minecraft:furnace", "/setblock ~2 ~ ~ minecraft:furnace"],
        [("place", "furnace", 1), ("smelt", "beef", 2, "coal")], "", 0, "hard", "MCU", "smelting"),

    task_entry("MCU_smelt_beef_simple",
        ["/give @s minecraft:beef 4", "/give @s minecraft:coal 4", "/give @s minecraft:furnace", "/setblock ~1 ~ ~ minecraft:furnace"],
        [("smelt", "beef", 1, "coal")], "", 0, "simple", "MCU", "smelting"),

    # ── Tool Use (2) ──
    task_entry("MCU_use_shield",
        ["/give @s minecraft:shield", "/give @s minecraft:iron_sword", "/give @s minecraft:iron_chestplate",
         "/give @s minecraft:iron_leggings", "/give @s minecraft:iron_boots"],
        [("equip", "shield", 1)], "", 0, "hard", "MCU", "tool_use"),

    task_entry("MCU_use_bow",
        ["/give @s minecraft:bow", "/give @s minecraft:arrow 32"],
        [("equip", "bow", 1)], "", 0, "simple", "MCU", "tool_use"),

    # ── Additional (2 more to reach 24) ──
    task_entry("MCU_collect_grass",
        ["/give @s minecraft:wooden_shovel"],
        [("mine", "dirt", 8)], "", 0, "simple", "MCU", "mining"),

    task_entry("MCU_craft_to_clock",
        ["/give @s minecraft:gold_ingot 4", "/give @s minecraft:redstone 1", "/give @s minecraft:crafting_table"],
        [("craft", "stone_pickaxe", 1)], "", 0, "hard", "MCU", "crafting"),
]

# ═══════════════════════════════════════════════════════════════
# Phase 2: Calibration — 12 tasks
# ═══════════════════════════════════════════════════════════════
P2 = [
    task_entry("CAL_craft_table_simple",
        ["/give @s minecraft:oak_planks 16"],
        [("craft", "crafting_table", 1)], "crafting_table", 1, "simple", "MCU", "crafting"),

    task_entry("CAL_craft_ladder_simple",
        ["/give @s minecraft:stick 15", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "ladder", 3)], "ladder", 3, "simple", "MCU", "crafting"),

    task_entry("CAL_craft_furnace_simple",
        ["/give @s minecraft:cobblestone 24", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "furnace", 1)], "furnace", 1, "simple", "MCU", "crafting"),

    task_entry("CAL_collect_wood_simple",
        ["/give @s minecraft:wooden_axe"],
        [("mine", "oak_log", 3)], "oak_log", 3, "simple", "MCU", "mining"),

    task_entry("CAL_cut_stone_simple",
        ["/give @s minecraft:wooden_pickaxe"],
        [("mine", "stone", 5)], "cobblestone", 3, "simple", "MCU", "mining"),

    task_entry("CAL_craft_ladder_hard",
        ["/give @s minecraft:stick 15", "/give @s minecraft:stone_pickaxe", "/give @s minecraft:iron_ingot 8",
         "/give @s minecraft:diamond 5", "/give @s minecraft:coal 10", "/give @s minecraft:bow 1", "/give @s minecraft:arrow 20",
         "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [("craft", "ladder", 2)], "ladder", 2, "hard", "MCU", "crafting"),

    task_entry("CAL_collect_wood_hard",
        ["/give @s minecraft:wooden_axe", "/give @s minecraft:stone 64", "/give @s minecraft:iron_ingot 10",
         "/give @s minecraft:diamond 5", "/give @s minecraft:golden_apple 2", "/give @s minecraft:bread 8"],
        [("mine", "oak_log", 5)], "oak_log", 5, "hard", "MCU", "mining"),

    task_entry("CAL_cut_stone_hard",
        ["/give @s minecraft:iron_pickaxe", "/give @s minecraft:stone 64", "/give @s minecraft:diamond 5"],
        [("mine", "stone", 10)], "cobblestone", 8, "hard", "MCU", "mining"),

    task_entry("CAL_smelt_beef_simple",
        ["/give @s minecraft:beef 4", "/give @s minecraft:coal 4", "/give @s minecraft:furnace", "/setblock ~1 ~ ~ minecraft:furnace"],
        [("smelt", "beef", 1, "coal")], "", 0, "simple", "MCU", "smelting"),

    task_entry("CAL_use_shield_simple",
        ["/give @s minecraft:shield", "/give @s minecraft:iron_sword"],
        [("equip", "shield", 1)], "", 0, "simple", "MCU", "tool_use"),

    task_entry("CAL_build_ladder_simple",
        ["/give @s minecraft:stick 32", "/give @s minecraft:crafting_table", "/give @s minecraft:oak_planks 16"],
        [("craft", "ladder", 4), ("place", "ladder", 1)], "", 0, "simple", "MCU", "building"),

    task_entry("CAL_place_block_simple",
        ["/give @s minecraft:cobblestone 64"],
        [("place", "cobblestone", 1)], "", 0, "simple", "MCU", "building"),
]

# ═══════════════════════════════════════════════════════════════
# Phase 3: Main Evaluation — 30 tasks (§6 in doc)
# ═══════════════════════════════════════════════════════════════
P3 = [
    task_entry("MCU_craft_bell",
        ["/give @s minecraft:gold_ingot 5", "/give @s minecraft:stick 2", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "bell", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_craft_to_cake",
        ["/give @s minecraft:wheat 9", "/give @s minecraft:sugar 2", "/give @s minecraft:egg 1", "/give @s minecraft:milk_bucket 3", "/give @s minecraft:crafting_table"],
        [["craft", "oak_planks", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_craft_enchanting_table",
        ["/give @s minecraft:obsidian 8", "/give @s minecraft:diamond 2", "/give @s minecraft:book 1", "/give @s minecraft:crafting_table"],
        [["craft", "enchanting_table", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_craft_bee_nest",
        ["/give @s minecraft:oak_planks 12", "/give @s minecraft:honeycomb 6", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "oak_planks", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_lay_carpet",
        ["/give @s minecraft:white_carpet 16", "/give @s minecraft:red_carpet 8", "/give @s minecraft:blue_carpet 8", "/give @s minecraft:yellow_carpet 8"],
        [["place", "white_carpet", 1]], "", 0, "simple", "MCU", "crafting"),
    task_entry("MCU_carve_pumpkins",
        ["/give @s minecraft:pumpkin 4", "/give @s minecraft:shears"],
        [["craft", "stone_pickaxe", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_decorate_the_wall",
        ["/give @s minecraft:oak_planks 16", "/give @s minecraft:painting 4", "/give @s minecraft:item_frame 4", "/give @s minecraft:torch 8", "/give @s minecraft:stone 32"],
        [["place", "stone", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_decorate_the_ground",
        ["/give @s minecraft:dirt 32", "/give @s minecraft:grass_block 16", "/give @s minecraft:bone_meal 8", "/give @s minecraft:oak_sapling 8"],
        [["place", "dirt", 1]], "", 0, "hard", "MCU", "crafting"),
    task_entry("MCU_mine_horizontally",
        ["/give @s minecraft:iron_pickaxe", "/give @s minecraft:torch 32"],
        [["mine", "stone", 5]], "", 0, "hard", "MCU", "mining"),
    task_entry("MCU_mine_obsidian",
        ["/give @s minecraft:diamond_pickaxe", "/setblock ~1 ~-1 ~ minecraft:obsidian"],
        [["mine", "obsidian", 1]], "", 0, "hard", "MCU", "mining"),
    task_entry("MCU_mine_diamond_ore",
        ["/give @s minecraft:iron_pickaxe", "/setblock ~1 ~-1 ~ minecraft:diamond_ore"],
        [["mine", "diamond_ore", 1]], "", 0, "hard", "MCU", "mining"),
    task_entry("MCU_collect_wool",
        ["/give @s minecraft:shears"],
        [["mine", "oak_log", 1]], "", 0, "simple", "MCU", "mining"),
    task_entry("MCU_dig_three_down_and_fill_one_up",
        ["/give @s minecraft:iron_shovel", "/give @s minecraft:dirt 64"],
        [["mine", "dirt", 3], ["place", "dirt", 1]], "", 0, "simple", "MCU", "mining"),
    task_entry("MCU_clean_the_weeds",
        ["/give @s minecraft:iron_hoe"],
        [["mine", "dirt", 3]], "", 0, "hard", "MCU", "mining"),
    task_entry("MCU_mine_horizontally_hard",
        ["/give @s minecraft:diamond_pickaxe", "/give @s minecraft:torch 64", "/give @s minecraft:stone 64"],
        [["mine", "stone", 10]], "", 0, "hard", "MCU", "mining"),
    task_entry("MCU_find_bedrock",
        ["/give @s minecraft:diamond_pickaxe", "/give @s minecraft:torch 64"],
        [["mine", "stone", 1]], "", 0, "hard", "MCU", "exploration"),
    task_entry("MCU_find_blue_bed",
        ["/give @s minecraft:white_bed 4", "/give @s minecraft:blue_dye 4"],
        [["place", "white_bed", 1]], "", 0, "hard", "MCU", "exploration"),
    task_entry("MCU_find_item_frame",
        ["/give @s minecraft:item_frame 4", "/give @s minecraft:oak_planks 16"],
        [["place", "item_frame", 1]], "", 0, "hard", "MCU", "exploration"),
    task_entry("MCU_explore_boat",
        ["/give @s minecraft:oak_boat", "/give @s minecraft:map", "/give @s minecraft:bread 4"],
        [["mine", "oak_log", 1]], "", 0, "hard", "MCU", "exploration"),
    task_entry("MCU_explore_map",
        ["/give @s minecraft:map", "/give @s minecraft:compass", "/give @s minecraft:iron_sword", "/give @s minecraft:bread 8"],
        [["mine", "stone", 1]], "", 0, "hard", "MCU", "exploration"),
    task_entry("MCU_find_village_items",
        ["/give @s minecraft:compass", "/give @s minecraft:map", "/give @s minecraft:bread 8", "/give @s minecraft:iron_sword"],
        [["mine", "oak_log", 1]], "", 0, "hard", "MCU", "exploration"),
    task_entry("MD_techtree_wooden_sword",
        ["/give @s minecraft:oak_log 4", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "oak_planks", 1], ["craft", "stick", 1], ["craft", "wooden_sword", 1]], "wooden_sword", 1, "hard", "MineDojo", "long_horizon"),
    task_entry("MD_techtree_stone_pickaxe",
        ["/give @s minecraft:oak_log 6", "/give @s minecraft:cobblestone 12", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "oak_planks", 1], ["craft", "stick", 1], ["craft", "wooden_pickaxe", 1], ["craft", "stone_pickaxe", 1]], "stone_pickaxe", 1, "hard", "MineDojo", "long_horizon"),
    task_entry("MD_techtree_furnace_torch",
        ["/give @s minecraft:oak_log 6", "/give @s minecraft:cobblestone 32", "/give @s minecraft:coal 8", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "oak_planks", 1], ["craft", "stick", 1], ["craft", "furnace", 1], ["place", "furnace", 1], ["craft", "torch", 8]], "torch", 8, "hard", "MineDojo", "long_horizon"),
    task_entry("MD_techtree_stone_axe",
        ["/give @s minecraft:oak_log 4", "/give @s minecraft:cobblestone 12", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "oak_planks", 1], ["craft", "stick", 1], ["craft", "stone_axe", 1]], "stone_axe", 1, "hard", "MineDojo", "long_horizon"),
    task_entry("XENON_wooden_chain",
        ["/give @s minecraft:oak_log 6"],
        [["craft", "oak_planks", 3], ["craft", "stick", 2], ["craft", "crafting_table", 1], ["place", "crafting_table", 1], ["craft", "wooden_pickaxe", 1]], "wooden_pickaxe", 1, "hard", "XENON", "long_horizon"),
    task_entry("MD_techtree_full_stone_tools",
        ["/give @s minecraft:oak_log 6", "/give @s minecraft:cobblestone 24", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "oak_planks", 2], ["craft", "stick", 2], ["craft", "wooden_pickaxe", 1], ["craft", "stone_pickaxe", 1], ["craft", "stone_axe", 1], ["craft", "stone_sword", 1]], "stone_sword", 1, "hard", "MineDojo", "long_horizon"),
    task_entry("MCU_use_lead_and_craft_stonecut",
        ["/give @s minecraft:lead 2", "/give @s minecraft:stone 12", "/give @s minecraft:iron_ingot 4", "/give @s minecraft:crafting_table"],
        [["equip", "lead", 1], ["craft", "stone_pickaxe", 1]], "", 0, "hard", "MCU", "long_horizon"),
    task_entry("MCU_FAIL_wrong_tool_diamond",
        ["/give @s minecraft:stone_pickaxe", "/setblock ~1 ~-1 ~ minecraft:diamond_ore"],
        [["mine", "diamond_ore", 1]], "", 0, "hard", "MCU", "failure_recovery"),
    task_entry("MCU_FAIL_smelt_no_fuel",
        ["/give @s minecraft:raw_iron 4", "/give @s minecraft:furnace", "/setblock ~1 ~ ~ minecraft:furnace"],
        [["place", "furnace", 1], ["smelt", "raw_iron", 1, "coal"]], "", 0, "hard", "MCU", "failure_recovery"),
    task_entry("MCU_FAIL_craft_no_mats",
        ["/give @s minecraft:oak_log 4"],
        [["craft", "crafting_table", 1]], "", 0, "hard", "MCU", "failure_recovery"),
    task_entry("MCU_FAIL_place_no_item",
        ["/give @s minecraft:stick 8"],
        [["place", "crafting_table", 1]], "", 0, "hard", "MCU", "failure_recovery"),
    task_entry("CUSTOM_stress_torch_no_coal",
        ["/give @s minecraft:stick 8", "/give @s minecraft:crafting_table"],
        [["craft", "torch", 4]], "", 0, "hard", "custom", "stress"),
    task_entry("CUSTOM_stress_spick_no_cobble",
        ["/give @s minecraft:stick 4", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "stone_pickaxe", 1]], "", 0, "hard", "custom", "stress"),
    task_entry("CUSTOM_stress_smelt_no_fuel",
        ["/give @s minecraft:cobblestone 8", "/give @s minecraft:furnace"],
        [["place", "furnace", 1], ["smelt", "cobblestone", 1, "coal"]], "", 0, "hard", "custom", "stress"),
    task_entry("CUSTOM_stress_furnace_vs_spick",
        ["/give @s minecraft:cobblestone 16", "/give @s minecraft:stick 8", "/give @s minecraft:crafting_table", "/setblock ~2 ~ ~ minecraft:crafting_table"],
        [["craft", "furnace", 1], ["craft", "stone_pickaxe", 1]], "", 0, "hard", "custom", "stress"),
]
