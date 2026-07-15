from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero import spaces

from typing import Dict, Any


class PlainInventoryObservation(TranslationHandler):
    """obs['plain_inventory']"""

    n_slots = 36

    def to_string(self) -> str:
        return "plain_inventory"

    def xml_template(self) -> str:
        return str("""<ObservationFromFullInventory flat="false"/>""")

    def __init__(self):
        shape = (self.n_slots,)
        space = spaces.Dict(
            {
                slot_id: spaces.Dict(
                    {
                        "type": spaces.Text(shape=()),
                        "quantity": spaces.Box(low=0, high=64, shape=()),
                    }
                )
                for slot_id in range(36)
            }
        )
        super().__init__(space=space)

    def from_hero(self, obs_dict: Dict[str, Any]):
        assert "inventory" in obs_dict, "Missing inventory key in malmo json"
        # print(obs_dict.keys())
        # print(obs_dict["inventory"]·)
        inventory = dict()
        for item in obs_dict["inventory"]:
            # Malmo 1.11.2 uses 'index', newer versions use 'slot_id'
            slot = item.get("slot_id", item.get("index", 0))
            inventory[slot] = {
                "type": item.get("type", ""),
                "quantity": item.get("quantity", 0),
            }

        return inventory
