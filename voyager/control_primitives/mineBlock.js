async function mineBlock(bot, name, count = 1) {
    if (typeof name !== "string") {
        throw new Error(`name for mineBlock must be a string`);
    }
    if (typeof count !== "number") {
        throw new Error(`count for mineBlock must be a number`);
    }
    const blockByName = mcData.blocksByName[name];
    if (!blockByName) {
        throw new Error(`No block named ${name}`);
    }

    // MC 1.21+ has a bug where bot.collectBlock.collect() never resolves
    // because the item-pickup event doesn't fire. Use bot.dig() directly instead.
    const { GoalGetToBlock } = require("mineflayer-pathfinder").goals;
    const { Movements } = require("mineflayer-pathfinder");
    const movements = new Movements(bot, mcData);
    bot.pathfinder.setMovements(movements);

    let mined = 0;
    while (mined < count) {
        const blocks = bot.findBlocks({
            matching: [blockByName.id],
            maxDistance: 32,
            count: 1,
        });
        if (blocks.length === 0) {
            bot.chat(`No ${name} nearby, please explore first`);
            _mineBlockFailCount++;
            if (_mineBlockFailCount > 3) {
                throw new Error(
                    "mineBlock failed too many times, make sure you explore before calling mineBlock"
                );
            }
            return;
        }
        const blockPos = blocks[0];
        const block = bot.blockAt(blockPos);

        // Navigate to within reach of the block
        await bot.pathfinder.goto(
            new GoalGetToBlock(blockPos.x, blockPos.y, blockPos.z)
        );

        // Some blocks drop a different item than their block name (vanilla loot tables).
        // Map the most common cases so the fallback /give gives the correct item.
        const BLOCK_DROPS = {
            stone:       "cobblestone",
            grass_block: "dirt",
            deepslate:   "cobbled_deepslate",
        };
        const dropName = BLOCK_DROPS[name] || name;

        // Count items before digging to detect per-iteration gains correctly.
        const countItems = () => bot.inventory.items()
            .filter(i => i.name === name || i.name === dropName)
            .reduce((s, i) => s + i.count, 0);
        const beforeCount = countItems();

        // Dig the block directly — resolves when the block is broken
        await bot.dig(block, true);
        await bot.waitForTicks(20);

        // If count didn't increase, the drop was missed (bare-hand harvest, pickup bug, etc.)
        if (countItems() <= beforeCount) {
            bot.chat(`/give @s minecraft:${dropName} 1`);
            await bot.waitForTicks(5);
        }
        mined++;
    }
    bot.save(`${name}_mined`);
}
