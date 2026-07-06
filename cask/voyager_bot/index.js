const fs = require("fs");
const express = require("express");
const bodyParser = require("body-parser");
const mineflayer = require("mineflayer");

const skills = require("./lib/skillLoader");
const { initCounter, getNextTime } = require("./lib/utils");
const obs = require("./lib/observation/base");
const OnChat = require("./lib/observation/onChat");
const OnError = require("./lib/observation/onError");
const { Voxels, BlockRecords } = require("./lib/observation/voxels");
const Status = require("./lib/observation/status");
const Inventory = require("./lib/observation/inventory");
const OnSave = require("./lib/observation/onSave");
const Chests = require("./lib/observation/chests");
const { plugin: tool } = require("mineflayer-tool");

let bot = null;

const app = express();

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: false }));

app.post("/start", (req, res) => {
    if (bot) onDisconnect("Restarting bot");
    bot = null;
    console.log(req.body);
    const crypto = require('crypto');
    const fs = require('fs');
    const botName = "cask_bot";
    // Pre-op the bot via ops.json (offline UUID computation)
    try {
        const md5 = crypto.createHash('md5').update('OfflinePlayer:' + botName, 'utf8').digest();
        md5[6] = (md5[6] & 0x0f) | 0x30;
        md5[8] = (md5[8] & 0x3f) | 0x80;
        const hex = md5.toString('hex');
        const uuid = hex.substring(0,8)+'-'+hex.substring(8,12)+'-'+hex.substring(12,16)+'-'+hex.substring(16,20)+'-'+hex.substring(20);
        fs.writeFileSync('D:/mc java/ops.json', JSON.stringify([{
            uuid, name: botName, level: 4, bypassesPlayerLimit: true
        }], null, 2));
        console.log("[op] pre-opped", botName, uuid);
    } catch(e) { console.log("[op] fail:", e.message); }
    bot = mineflayer.createBot({
        host: "localhost", // minecraft server ip
        port: req.body.port, // minecraft server port
        username: botName,
        disableChatSigning: true,
        checkTimeoutInterval: 60 * 60 * 1000,
    });
    bot.once("error", onConnectionFailed);

    // ──── Chat fix: only on 1.21+ (prismarine-chat crash, chat_command packet) ────
    bot.once('spawn', async () => {
        // Check if bot version requires chat fix (1.21+ = prismarine-chat crash)
        const isNewChat = bot.supportFeature && bot.supportFeature('useChatSessions');
        if (!isNewChat) {
            console.log("[chat] 1.20.x detected, native bot.chat works");
            return;
        }
        console.log("[chat] applying 1.21+ chat fix");
        bot.chat = (message) => {
            try {
                if (message.startsWith('/')) {
                    bot._client.write('chat_command', {
                        command: message.slice(1),
                        timestamp: BigInt(Date.now()),
                        salt: 1n,
                        argumentSignatures: [],
                        messageCount: 0,
                        acknowledged: Buffer.alloc(3, 0)
                    });
                } else {
                    bot._client.write('chat', { message });
                }
            } catch (e) {
                console.log("[chat.send]", e.message);
            }
        };
        const pcl = bot._client.listeners('playerChat');
        const scl = bot._client.listeners('systemChat');
        pcl.forEach(l => bot._client.removeListener('playerChat', l));
        scl.forEach(l => bot._client.removeListener('systemChat', l));
        bot._client.on('playerChat', (data) => {
            try { bot.emit('messagestr', String(data.plainMessage || data.formattedMessage || ''), 'chat', null, data.sender, false); }
            catch (e) { /* ignore */ }
        });
        bot._client.on('systemChat', (data) => {
            try { bot.emit('messagestr', String(data.formattedMessage || ''), 'system', null); }
            catch (e) { /* ignore */ }
        });
    });
    // ──── End chat fix ────

    // Event subscriptions
    bot.waitTicks = req.body.waitTicks;
    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];
    bot.iron_pickaxe = false;

    bot.on("kicked", onDisconnect);

    // mounting will cause physicsTick to stop
    bot.on("mount", () => {
        bot.dismount();
    });

    bot.once("spawn", async () => {
        bot.removeListener("error", onConnectionFailed);
        let itemTicks = 1;
        if (req.body.reset === "hard") {
            bot.chat("/clear @s");
            bot.chat("/kill @s");
            const inventory = req.body.inventory ? req.body.inventory : {};
            const equipment = req.body.equipment
                ? req.body.equipment
                : [null, null, null, null, null, null];
            for (let key in inventory) {
                bot.chat(`/give @s minecraft:${key} ${inventory[key]}`);
                itemTicks += 1;
            }
            const equipmentNames = [
                "armor.head",
                "armor.chest",
                "armor.legs",
                "armor.feet",
                "weapon.mainhand",
                "weapon.offhand",
            ];
            for (let i = 0; i < 6; i++) {
                if (i === 4) continue;
                if (equipment[i]) {
                    bot.chat(
                        `/item replace entity @s ${equipmentNames[i]} with minecraft:${equipment[i]}`
                    );
                    itemTicks += 1;
                }
            }
        }

        if (req.body.position) {
            bot.chat(
                `/tp @s ${req.body.position.x} ${req.body.position.y} ${req.body.position.z}`
            );
        }

        // if iron_pickaxe is in bot's inventory
        if (
            bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.iron_pickaxe = true;
        }

        const { pathfinder } = require("mineflayer-pathfinder");
        const tool = require("mineflayer-tool").plugin;
        const collectBlock = require("mineflayer-collectblock").plugin;
        const pvp = require("mineflayer-pvp").plugin;
        const hawkeye = require("minecrafthawkeye").default;
        bot.loadPlugin(pathfinder);
        bot.loadPlugin(tool);
        bot.loadPlugin(collectBlock);
        bot.loadPlugin(pvp);
        bot.loadPlugin(hawkeye);

        // bot.collectBlock.movements.digCost = 0;
        // bot.collectBlock.movements.placeCost = 0;

        obs.inject(bot, [
            OnChat,
            OnError,
            Voxels,
            Status,
            Inventory,
            OnSave,
            Chests,
            BlockRecords,
        ]);
        skills.inject(bot);

        if (req.body.spread) {
            bot.chat(`/spreadplayers ~ ~ 0 300 under 80 false @s`);
            await bot.waitForTicks(bot.waitTicks);
        }

        await bot.waitForTicks(bot.waitTicks * itemTicks);
        res.json(bot.observe());

        initCounter(bot);
        bot.chat("/gamerule keepInventory true");
        bot.chat("/gamerule doDaylightCycle false");
    });

    function onConnectionFailed(e) {
        console.log(e);
        bot = null;
        res.status(400).json({ error: e });
    }
    function onDisconnect(message) {
        if (bot.viewer) {
            bot.viewer.close();
        }
        bot.end();
        console.log(message);
        bot = null;
    }
});

app.post("/action", async (req, res) => {
    // CASKe: structured action endpoint.
    // body: { action: "mine"|"craft"|"smelt"|"place"|"kill"|"equip"|"inventory", target?, count?, observe? }
    if (!bot) return res.status(400).json({ success: false, error: "bot not started" });
    if (!global.mcData) global.mcData = require("minecraft-data")(bot.version);
    const md = global.mcData;
    if (!global.Vec3) global.Vec3 = require("vec3").Vec3;
    const { Vec3 } = global;
    const { Movements: Moves, goals: { GoalNear, GoalLookAtBlock, GoalPlaceBlock } } = require("mineflayer-pathfinder");
    if (!global.GoalLookAtBlock) global.GoalLookAtBlock = GoalLookAtBlock;
    if (!global.GoalPlaceBlock) global.GoalPlaceBlock = GoalPlaceBlock;
    const moves = new Moves(bot, md);
    bot.pathfinder.setMovements(moves);
    // Init fail counters (accessed as implicit globals by control primitives)
    if (global._mineBlockFailCount === undefined) { global._mineBlockFailCount = 0; global._craftItemFailCount = 0;
        global._smeltItemFailCount = 0; global._killMobFailCount = 0; global._placeItemFailCount = 0; }

    // --- Stuck detection (no teleport — causes anti-cheat kicks) ---
    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];
    function onPhysicsTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
        }
    }
    bot.on("physicsTick", onPhysicsTick);

    const a = req.body, act = a.action, tgt = a.target || "", cnt = a.count || 1;
    let result = { success: false }, got = 0;

    try {
        if (act === "mine") {
            await require("./control_primitives/mineBlock.js").mineBlock(bot, tgt || "oak_log", cnt);
            got = bot.inventory.items().filter(i => (i.name||"").includes(tgt||"log")).reduce((s,i)=>s+i.count,0);
            result = { success: got >= cnt, details: `${tgt||"log"}:${got}` };
        } else if (act === "craft") {
            await require("./control_primitives/craftItem.js").craftItem(bot, tgt, cnt);
            result = { success: true };
        } else if (act === "smelt") {
            const fuel = a.fuel || "coal";
            await require("./control_primitives/smeltItem.js").smeltItem(bot, tgt, fuel, cnt);
            result = { success: true };
        } else if (act === "place") {
            const pos = a.position
                ? new Vec3(a.position.x, a.position.y, a.position.z)
                : new Vec3(Math.floor(bot.entity.position.x) + 1, Math.floor(bot.entity.position.y), Math.floor(bot.entity.position.z));  // default: next to bot
            await require("./control_primitives/placeItem.js")(bot, tgt, pos);
            result = { success: true };
        } else if (act === "kill") {
            await require("./control_primitives/killMob.js")(bot, tgt, a.timeout || 60);
            result = { success: true };
        } else if (act === "equip") {
            const item = bot.inventory.items().find(i => (i.name || '').includes(tgt));
            if (!item) result = { success: false, error: `no ${tgt}` };
            else { await bot.equip(item, "hand"); result = { success: true, details: `equipped ${tgt}` }; }
        } else if (act === "inventory") {
            result = { success: true, details: JSON.stringify(bot.inventory.items().map(i=>({name:i.name,count:i.count}))) };
        } else if (act === "command") {
            // Execute arbitrary chat command (for /give, /setblock, etc.)
            bot.chat(tgt);
            result = { success: true, details: tgt };
        } else {
            result = { success: false, error: "unknown: " + act };
        }
    } catch (e) {
        if (act === "mine") {
            got = bot.inventory.items().filter(i => (i.name||"").includes(tgt||"log")).reduce((s,i)=>s+i.count,0);
            result = { success: got >= cnt, details: `${tgt||"log"}:${got}` };
        } else { result = { success: false, error: e.message?.slice(0,80) }; }
    }

    bot.removeListener("physicsTick", onPhysicsTick);

    // Append observation state if requested (for XENON planner)
    if (a.observe !== false) {
        result.observe = {
            inventory: bot.inventory.items().reduce((acc, i) => { acc[i.name] = (acc[i.name]||0) + i.count; return acc; }, {}),
            status: { health: Math.floor(bot.health), food: bot.food, position: { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z } }
        };
    }
    res.json(result);
});

app.post("/step", async (req, res) => {
    // import useful package
    let response_sent = false;
    function otherError(err) {
        console.log("Uncaught Error");
        bot.emit("error", handleError(err));
        bot.waitForTicks(bot.waitTicks).then(() => {
            if (!response_sent) {
                response_sent = true;
                res.json(bot.observe());
            }
        });
    }

    process.on("uncaughtException", otherError);

    const mcData = require("minecraft-data")(bot.version);
    mcData.itemsByName["leather_cap"] = mcData.itemsByName["leather_helmet"];
    mcData.itemsByName["leather_tunic"] =
        mcData.itemsByName["leather_chestplate"];
    mcData.itemsByName["leather_pants"] =
        mcData.itemsByName["leather_leggings"];
    mcData.itemsByName["leather_boots"] = mcData.itemsByName["leather_boots"];
    mcData.itemsByName["lapis_lazuli_ore"] = mcData.itemsByName["lapis_ore"];
    mcData.blocksByName["lapis_lazuli_ore"] = mcData.blocksByName["lapis_ore"];
    const {
        Movements,
        goals: {
            Goal,
            GoalBlock,
            GoalNear,
            GoalXZ,
            GoalNearXZ,
            GoalY,
            GoalGetToBlock,
            GoalLookAtBlock,
            GoalBreakBlock,
            GoalCompositeAny,
            GoalCompositeAll,
            GoalInvert,
            GoalFollow,
            GoalPlaceBlock,
        },
        pathfinder,
        Move,
        ComputedPath,
        PartiallyComputedPath,
        XZCoordinates,
        XYZCoordinates,
        SafeBlock,
        GoalPlaceBlockOptions,
    } = require("mineflayer-pathfinder");
    const { Vec3 } = require("vec3");

    // Set up pathfinder
    const movements = new Movements(bot, mcData);
    bot.pathfinder.setMovements(movements);

    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];

    function onTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
            if (bot.stuckTickCounter >= 100) {
                onStuck(1.5);
                bot.stuckTickCounter = 0;
            }
        }
    }

    bot.on("physicsTick", onTick);

    // initialize fail count
    let _craftItemFailCount = 0;
    let _killMobFailCount = 0;
    let _mineBlockFailCount = 0;
    let _placeItemFailCount = 0;
    let _smeltItemFailCount = 0;

    // Retrieve array form post bod
    const code = req.body.code;
    const programs = req.body.programs;
    bot.cumulativeObs = [];
    await bot.waitForTicks(bot.waitTicks);
    const r = await evaluateCode(code, programs);
    process.off("uncaughtException", otherError);
    if (r !== "success") {
        bot.emit("error", handleError(r));
    }
    await returnItems();
    // wait for last message
    await bot.waitForTicks(bot.waitTicks);
    if (!response_sent) {
        response_sent = true;
        res.json(bot.observe());
    }
    bot.removeListener("physicTick", onTick);

    async function evaluateCode(code, programs) {
        // Echo the code produced for players to see it. Don't echo when the bot code is already producing dialog or it will double echo
        try {
            await eval("(async () => {" + programs + "\n" + code + "})()");
            return "success";
        } catch (err) {
            return err;
        }
    }

    function onStuck(posThreshold) {
        const currentPos = bot.entity.position;
        bot.stuckPosList.push(currentPos);

        // Check if the list is full
        if (bot.stuckPosList.length === 5) {
            const oldestPos = bot.stuckPosList[0];
            const posDifference = currentPos.distanceTo(oldestPos);

            if (posDifference < posThreshold) {
                teleportBot(); // execute the function
            }

            // Remove the oldest time from the list
            bot.stuckPosList.shift();
        }
    }

    function teleportBot() {
        const blocks = bot.findBlocks({
            matching: (block) => {
                return block.type === 0;
            },
            maxDistance: 1,
            count: 27,
        });

        if (blocks) {
            // console.log(blocks.length);
            const randomIndex = Math.floor(Math.random() * blocks.length);
            const block = blocks[randomIndex];
            bot.chat(`/tp @s ${block.x} ${block.y} ${block.z}`);
        } else {
            bot.chat("/tp @s ~ ~1.25 ~");
        }
    }

    function returnItems() {
        bot.chat("/gamerule doTileDrops false");
        const crafting_table = bot.findBlock({
            matching: mcData.blocksByName.crafting_table.id,
            maxDistance: 128,
        });
        if (crafting_table) {
            bot.chat(
                `/setblock ${crafting_table.position.x} ${crafting_table.position.y} ${crafting_table.position.z} air destroy`
            );
            bot.chat("/give @s crafting_table");
        }
        const furnace = bot.findBlock({
            matching: mcData.blocksByName.furnace.id,
            maxDistance: 128,
        });
        if (furnace) {
            bot.chat(
                `/setblock ${furnace.position.x} ${furnace.position.y} ${furnace.position.z} air destroy`
            );
            bot.chat("/give @s furnace");
        }
        if (bot.inventoryUsed() >= 32) {
            // if chest is not in bot's inventory
            if (!bot.inventory.items().find((item) => item.name === "chest")) {
                bot.chat("/give @s chest");
            }
        }
        // if iron_pickaxe not in bot's inventory and bot.iron_pickaxe
        if (
            bot.iron_pickaxe &&
            !bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.chat("/give @s iron_pickaxe");
        }
        bot.chat("/gamerule doTileDrops true");
    }

    function handleError(err) {
        let stack = err.stack;
        if (!stack) {
            return err;
        }
        console.log(stack);
        const final_line = stack.split("\n")[1];
        const regex = /<anonymous>:(\d+):\d+\)/;

        const programs_length = programs.split("\n").length;
        let match_line = null;
        for (const line of stack.split("\n")) {
            const match = regex.exec(line);
            if (match) {
                const line_num = parseInt(match[1]);
                if (line_num >= programs_length) {
                    match_line = line_num - programs_length;
                    break;
                }
            }
        }
        if (!match_line) {
            return err.message;
        }
        let f_line = final_line.match(
            /\((?<file>.*):(?<line>\d+):(?<pos>\d+)\)/
        );
        if (f_line && f_line.groups && fs.existsSync(f_line.groups.file)) {
            const { file, line, pos } = f_line.groups;
            const f = fs.readFileSync(file, "utf8").split("\n");
            // let filename = file.match(/(?<=node_modules\\)(.*)/)[1];
            let source = file + `:${line}\n${f[line - 1].trim()}\n `;

            const code_source =
                "at " +
                code.split("\n")[match_line - 1].trim() +
                " in your code";
            return source + err.message + "\n" + code_source;
        } else if (
            f_line &&
            f_line.groups &&
            f_line.groups.file.includes("<anonymous>")
        ) {
            const { file, line, pos } = f_line.groups;
            let source =
                "Your code" +
                `:${match_line}\n${code.split("\n")[match_line - 1].trim()}\n `;
            let code_source = "";
            if (line < programs_length) {
                source =
                    "In your program code: " +
                    programs.split("\n")[line - 1].trim() +
                    "\n";
                code_source = `at line ${match_line}:${code
                    .split("\n")
                    [match_line - 1].trim()} in your code`;
            }
            return source + err.message + "\n" + code_source;
        }
        return err.message;
    }
});

app.post("/stop", (req, res) => {
    if (bot) bot.end();
    bot = null;
    res.json({ message: "Bot stopped" });
});

app.post("/pause", (req, res) => {
    if (!bot) {
        res.status(400).json({ error: "Bot not spawned" });
        return;
    }
    bot.chat("/pause");
    bot.waitForTicks(bot.waitTicks).then(() => {
        res.json({ message: "Success" });
    });
});

// Server listening to PORT 3000

// Error handling so one bad request doesn't kill the server
process.on("uncaughtException", (e) => console.error("UNCAUGHT:", e.message));
process.on("unhandledRejection", (e) => console.error("UNHANDLED:", e.message));

const DEFAULT_PORT = 3000;
const PORT = process.argv[2] || DEFAULT_PORT;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});
