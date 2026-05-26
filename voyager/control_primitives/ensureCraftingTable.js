async function ensureCraftingTable(bot) {
    // 1. Already a crafting table nearby — nothing to do
    const existing = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 8,
    });
    if (existing) {
        bot.chat("Crafting table already nearby");
        return existing.position;
    }

    // 2. Not in inventory — craft one
    const tableItem = bot.inventory.findInventoryItem(
        mcData.itemsByName.crafting_table.id
    );
    if (!tableItem) {
        // Count all plank types we already have
        const plankTypes = [
            "oak_planks", "spruce_planks", "birch_planks",
            "jungle_planks", "acacia_planks", "dark_oak_planks",
            "mangrove_planks", "cherry_planks", "bamboo_planks",
        ];
        let totalPlanks = 0;
        let availablePlankName = null;
        for (const p of plankTypes) {
            const entry = mcData.itemsByName[p];
            if (!entry) continue;
            const item = bot.inventory.findInventoryItem(entry.id);
            if (item && item.count > 0) {
                totalPlanks += item.count;
                if (!availablePlankName) availablePlankName = p;
            }
        }

        // Need at least 4 planks for a crafting table
        if (totalPlanks < 4) {
            bot.chat("Not enough planks, mining a log first");
            // Try oak_log; if not present the bot should have explored first
            const logTypes = [
                "oak_log", "spruce_log", "birch_log", "jungle_log",
                "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
            ];
            let logName = null;
            for (const l of logTypes) {
                if (bot.findBlock({
                    matching: mcData.blocksByName[l]?.id,
                    maxDistance: 32,
                })) { logName = l; break; }
            }
            if (!logName) logName = "oak_log"; // fallback; mineBlock will fail with a clear message
            await mineBlock(bot, logName, 1);

            // Determine plank type from log type
            const plankName = logName.replace("_log", "_planks");
            await craftItem(bot, plankName, 1); // yields 4 planks
            availablePlankName = plankName;
        }

        bot.chat("Crafting a crafting table");
        // craftItem needs a 2x2 recipe — no table needed
        await craftItem(bot, "crafting_table", 1);
    }

    // 3. Place it adjacent to the bot
    const targetPos = bot.entity.position.floored().offset(1, 0, 0);
    bot.chat(`Placing crafting table at ${targetPos}`);
    await placeItem(bot, "crafting_table", targetPos);
    await bot.waitForTicks(5);

    const placed = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 8,
    });
    if (!placed) {
        throw new Error("ensureCraftingTable: placement failed — no table within 8 blocks");
    }
    bot.chat("Crafting table ready");
    return placed.position;
}
