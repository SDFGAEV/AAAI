async function craftItem(bot, name, count = 1) {
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error("name for craftItem must be a string");
    }
    // return if count is not number
    if (typeof count !== "number") {
        throw new Error("count for craftItem must be a number");
    }
    const failedCraftFeedback = require("./craftHelper.js");
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }
    const craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 16,
    });
    if (!craftingTable) {
        // bot.chat("Craft without a crafting table");
    } else {
        await bot.pathfinder.goto(
            new GoalLookAtBlock(craftingTable.position, bot.world)
        );
    }
    // mcData.recipes[itemId] — works on all MC versions
    const recipeVariants = mcData.recipes[itemByName.id] || [];
    if (recipeVariants.length === 0) throw new Error(`no mc-data recipe for ${name}`);
    const recipeData = recipeVariants[0];  // first variant

    // Build Recipe object from mcData format using prismarine-recipe
    const { Recipe } = require('prismarine-recipe')(bot.version);
    const recipe = new Recipe(recipeData);
    const before = bot.inventory.items()
        .filter(i => i.type === itemByName.id).reduce((s,i) => s + i.count, 0);
    try {
        await bot.craft(recipe, count, craftingTable);
        await bot.waitForTicks(20);
    } catch (err) {
        throw new Error(`craft ${name} failed: ${err.message}`);
    }
    // Verify item count increased
    const after = bot.inventory.items()
        .filter(i => i.type === itemByName.id).reduce((s,i) => s + i.count, 0);
    if (after <= before) {
        _craftItemFailCount++;
        if (_craftItemFailCount >= 3) {
            throw new Error(`craft ${name}:${count}x failed (count ${before}→${after})`);
        }
    }
}


module.exports = craftItem;


module.exports = craftItem;


module.exports = { craftItem };
