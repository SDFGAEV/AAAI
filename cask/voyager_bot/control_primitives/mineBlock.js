async function mineBlock(bot, name, count = 1) {
    if (typeof name !== "string") throw new Error(`name for mineBlock must be a string`);
    if (typeof count !== "number") throw new Error(`count for mineBlock must be a number`);
    const blockByName = mcData.blocksByName[name];
    if (!blockByName) throw new Error(`No block named ${name}`);

    // Wait for chunks to load
    for (let i = 0; i < 30; i++) {
        const b = bot.blockAt(bot.entity.position.offset(0, -1, 0));
        if (b && b.name !== 'void_air' && b.name !== 'air') break;
        await bot.waitForTicks(10);
    }

    const blocks = bot.findBlocks({
        matching: [blockByName.id],
        maxDistance: 16,
        count: 256,
    });
    if (blocks.length === 0) throw new Error(`No ${name} nearby`);
    const targets = blocks.map(p => bot.blockAt(p));

    await bot.collectBlock.collect(targets, {
        ignoreNoPath: true,
        count: count,
    });

    const got = bot.inventory.items()
        .filter(i => i.name === name || (i.name || '').includes(name.split('_')[0]))
        .reduce((s, i) => s + i.count, 0);
    if (got < count) throw new Error(`mine ${name}: only got ${got}/${count}`);
    bot.save(`${name}_mined`);
}


module.exports = mineBlock;


module.exports = mineBlock;


module.exports = { mineBlock };
