from __future__ import annotations

import json

from .schemas import StructuredAction


class ActionRenderer:
    """Turns a structured action into executable Mineflayer JS."""

    def render(self, action: StructuredAction) -> str:
        if action.type == "mine":
            return (
                f"bot.chat('EvaluativeController: mine {action.target} x{action.count}');\n"
                f"await mineBlock(bot, {json.dumps(action.target)}, {action.count});"
            )
        if action.type == "craft":
            return (
                f"bot.chat('EvaluativeController: craft {action.target} x{action.count}');\n"
                f"await craftItem(bot, {json.dumps(action.target)}, {action.count});"
            )
        if action.type == "noop":
            return "bot.chat('EvaluativeController: noop');\nawait bot.waitForTicks(20);"
        raise ValueError(f"Unsupported structured action: {action}")
