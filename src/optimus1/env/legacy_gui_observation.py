"""GUI observation compatible with both MCP-Reborn and legacy Malmo.

Legacy Malmo 0.37 (Minecraft 1.11.2) rejects ``<IsGuiOpen/>`` in Mission.xsd.
The handler therefore keeps the official observation key in the Gym space while
omitting that XML element unless the native implementation is explicitly
enabled.  This makes the state schema stable without claiming a GUI signal that
legacy Malmo cannot provide.
"""
import os
from typing import Any, Dict

from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler


class LegacyCompatibleIsGuiOpen(TranslationHandler):
    def __init__(self) -> None:
        super().__init__(spaces.Discrete(2))
        self._native = os.getenv("CACT_NATIVE_GUI_OBSERVATION", "0") == "1"

    def to_string(self) -> str:
        return "isGuiOpen"

    def xml_template(self) -> str:
        return "<IsGuiOpen/>" if self._native else ""

    def from_hero(self, obs: Dict[str, Any]) -> int:
        return int(bool(obs.get("isGuiOpen", False)))

    def from_universal(self, obs: Dict[str, Any]) -> int:
        return int(bool(obs.get("isGuiOpen", False)))

    def to_hero(self, x: Any) -> str:
        raise NotImplementedError("isGuiOpen is an observation-only handler")