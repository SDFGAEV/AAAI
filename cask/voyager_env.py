"""
VoyagerEnv: drop-in replacement for XENON's MineRL env.

Replaces env.get_status(), check_original_goal_finish(), etc.
with HTTP calls to the Voyager Mineflayer server.
"""

import os, json, logging, threading
import requests as _requests

_VOYAGER_PORT = int(os.environ.get("VOYAGER_PORT", "3000"))
_VOYAGER_URL = f"http://localhost:{_VOYAGER_PORT}"


class _DummyThread:
    """Mimics the thread returned by env.save_video()."""
    def join(self, timeout=None):
        pass
    def get_result(self):
        return None


class VoyagerEnv:
    """
    Adapter that exposes the same interface as XENON's CustomEnvWrapper
    but talks to the Voyager Mineflayer bot over HTTP.
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("VoyagerEnv")
        self.timeout = 1800
        self.num_steps = 0
        self.current_subgoal_finish = False
        self._prev_inv = {}
        self._cur_inv = {}
        self.api_thread = None
        self.can_change_hotbar = True
        self.can_open_inventory = True

    # ---- env interface used by XENON planner ----

    def reset(self):
        """Return minimal obs dict (planner may use obs["pov"] as image)."""
        # Voyager doesn't provide POV images; planner falls back to text context
        self._update_inventory()
        return {"pov": None}

    def get_status(self) -> dict:
        """
        Returns { "inventory": {...}, "location_stats": {...} }
        by calling Voyager /action with action=inventory + observe.
        """
        try:
            r = _requests.post(f"{_VOYAGER_URL}/action",
                               json={"action": "inventory", "observe": True},
                               timeout=15)
            data = r.json()
            obs = data.get("observe", {})
            inv = obs.get("inventory", {})
            pos = obs.get("status", {}).get("position", {"x": 0, "y": 0, "z": 0})
            self._prev_inv = self._cur_inv
            self._cur_inv = inv
            return {
                "inventory": inv,
                "location_stats": {
                    "xpos": pos["x"],
                    "ypos": pos["y"],
                    "zpos": pos["z"],
                }
            }
        except Exception as e:
            self.logger.warning(f"VoyagerEnv.get_status() failed: {e}")
            return {"inventory": {}, "location_stats": {"xpos": 0, "ypos": 0, "zpos": 0}}

    def check_original_goal_finish(self, goal_pair) -> bool:
        """
        Check if the final goal item is in inventory.
        goal_pair = [item_name, count]
        """
        if not goal_pair or len(goal_pair) < 1:
            return False
        goal_item = goal_pair[0].lower()
        need = int(goal_pair[1]) if len(goal_pair) > 1 else 1
        # Refresh inventory
        st = self.get_status()
        inv = st.get("inventory", {})
        for name, count in inv.items():
            if goal_item in name.lower() and count >= need:
                return True
        return False

    def save_video(self, *args, **kwargs):
        """No-op: Voyager runs headless."""
        return _DummyThread()

    def close(self):
        """No-op: Voyager bot stays alive."""
        pass

    def noop_action(self):
        """Stub for XENON NewHelper compatibility."""
        return {}

    def step(self, *args, **kwargs):
        """Stub for XENON NewHelper compatibility."""
        return None, 0, False, {}

    @property
    def inventory_new_item(self) -> bool:
        return len(self._cur_inv) > len(self._prev_inv)

    def inventory_new_item_what(self) -> dict:
        return {k: v for k, v in self._cur_inv.items()
                if k not in self._prev_inv}

    # ---- unused stubs (satisfy XENON's env interface) ----
    instances = []

    def api_thread_is_alive(self):
        return False

    # ---- internal ----

    def _update_inventory(self):
        try:
            r = _requests.post(f"{_VOYAGER_URL}/action",
                               json={"action": "inventory", "observe": False},
                               timeout=10)
            data = r.json()
            raw = json.loads(data.get("details", "[]"))
            inv = {}
            for item in raw:
                inv[item["name"]] = inv.get(item["name"], 0) + item["count"]
            self._prev_inv = self._cur_inv
            self._cur_inv = inv
        except Exception:
            pass
