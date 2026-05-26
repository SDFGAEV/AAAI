"""
Targeted test for placeItem.js blockUpdate-timeout fix.
Gives the bot a crafting_table via /give, then tries placeItem at several positions.
Does not use the full Voyager agent stack — only the env bridge.
"""
import os, sys
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

sys.path.insert(0, ".")
from voyager.env import VoyagerEnv
from voyager.control_primitives import load_control_primitives

env = VoyagerEnv(mc_port=55916, server_port=3000, request_timeout=60)
primitives = "\n\n".join(load_control_primitives())

# Hard reset → clear inventory
print("Resetting environment...")
env.reset(options={"mode": "hard", "wait_ticks": 20})

# Give bot a crafting_table
print("Giving bot crafting_table via /give...")
events = env.step("bot.chat('/give @s minecraft:crafting_table 1');")
events = env.step("await bot.waitForTicks(10);")

# Check inventory
check = env.step("""
const inv = bot.inventory.items().map(i => i.name + ':' + i.count).join(', ');
bot.chat('Inventory: ' + (inv || 'empty'));
""")
inv_str = [e[1].get("inventory", {}) for e in check if e[0] == "observe"]
print("Inventory:", inv_str[-1] if inv_str else "unknown")

# Test placeItem at bot position offset +1 on X axis
print("\nTesting placeItem (should trigger blockUpdate path)...")
test_code = """
const pos = bot.entity.position.floored().offset(1, 0, 0);
bot.chat('Attempting placeItem at ' + pos);
try {
    await placeItem(bot, "crafting_table", pos);
} catch(e) {
    bot.chat('placeItem threw: ' + e.message);
}

// Verify 1: block appeared at expected position?
const placed = bot.blockAt(pos);
bot.chat('Block at target pos: ' + (placed ? placed.name : 'null'));

// Verify 2: craftItem can find and use the crafting table?
await bot.waitForTicks(5);
const tableNearby = bot.findBlock({
    matching: mcData.blocksByName.crafting_table.id,
    maxDistance: 8
});
bot.chat('craftItem can find table: ' + (tableNearby ? 'yes at ' + tableNearby.position : 'no'));

// Verify 3: try to craft something with it
if (tableNearby) {
    await craftItem(bot, "oak_planks", 1);
}

const inv = bot.inventory.items().map(i => i.name + ':' + i.count).join(', ');
bot.chat('Final inventory: ' + (inv || 'empty'));
"""

result_events = env.step(test_code, programs=primitives)

print("\nMC Chat log from test:")
for etype, edata in result_events:
    if etype == "onChat":
        print(" ", edata.get("onChat", ""))
    elif etype == "observe":
        inv = edata.get("inventory", {})
        voxels = edata.get("voxels", [])
        print(f"  [observe] inventory={inv}, voxels_near={voxels[:5]}")

# Summary
placed_in_world = any(
    "crafting_table" in e[1].get("voxels", [])
    for e in result_events if e[0] == "observe"
)
table_in_inv = any(
    "crafting_table" in e[1].get("inventory", {})
    for e in result_events if e[0] == "observe"
)
print(f"\nResult: crafting_table placed in world? {placed_in_world}")
print(f"Result: crafting_table still in inventory? {table_in_inv}")

if placed_in_world:
    print("✓ placeItem fix works — table placed despite potential blockUpdate timeout")
elif not table_in_inv and not placed_in_world:
    print("? Table consumed but not found in voxels (might be placed just outside 8-block scan)")
else:
    print("✗ Table still in inventory and not in voxels — placement failed")

env.close()
