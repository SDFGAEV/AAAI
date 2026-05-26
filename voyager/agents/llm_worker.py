from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from voyager.prompts import load_prompt
from voyager.utils.json_utils import fix_and_parse_json
from .value_matrix import ValueMatrix
from .constraint_engine import Constraint, ConstraintEngine
from .entity_registry import EntityRegistry
from .task_dag import DAGNode, TaskDAG


class LLMWorker:
    """
    Synchronous background planning agent.

    Calls the LLM once per invocation to produce a structured snapshot:
      V – value matrix scores (updated into ValueMatrix)
      C – constraint set      (updated into ConstraintEngine)
      G – task DAG            (returned as TaskDAG)

    All subsequent Controller decisions are made locally against these snapshots
    without any further LLM calls. This keeps LLM out of the real-time loop.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        request_timeout: int = 60,
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
        )
        self._system_prompt = load_prompt("llm_worker")

    # ── high-level entry point ────────────────────────────────────────────────

    def update_from_events(
        self,
        events: list,
        task: str,
        completed_tasks: list,
        value_matrix: ValueMatrix,
        constraint_engine: ConstraintEngine,
        entity_registry: EntityRegistry,
        max_retries: int = 3,
    ) -> TaskDAG | None:
        """
        Parse events, call LLM once, update all three data structures in-place.
        Returns a fresh TaskDAG (G), or None if the LLM call fails.

        entity_registry is updated locally from events before the LLM call —
        no LLM needed for entity tracking.
        """
        entity_registry.update_from_observation(events)

        state = self._extract_state(events)
        if state is None:
            print("\033[33m[LLMWorker] No observe event found in events list.\033[0m")
            return None

        snapshot = self._call_llm(state, task, completed_tasks, max_retries)
        if snapshot is None:
            return None

        if "V" in snapshot and isinstance(snapshot["V"], dict):
            value_matrix.replace_all(snapshot["V"])
            print("\033[36m[LLMWorker] ValueMatrix updated.\033[0m")

        if "C" in snapshot and isinstance(snapshot["C"], list):
            try:
                constraints = [Constraint(**c) for c in snapshot["C"]]
                constraint_engine.replace_all(constraints)
                print(f"\033[36m[LLMWorker] ConstraintEngine updated ({len(constraints)} constraints).\033[0m")
            except Exception as e:
                print(f"\033[33m[LLMWorker] Failed to parse constraints: {e}\033[0m")

        if "G" in snapshot and isinstance(snapshot["G"], dict):
            try:
                dag = TaskDAG.from_dict(snapshot["G"])
                print(f"\033[36m[LLMWorker] TaskDAG built: {dag}\033[0m")
                return dag
            except Exception as e:
                print(f"\033[33m[LLMWorker] Failed to parse TaskDAG: {e}\033[0m")

        return None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _call_llm(
        self,
        state: dict,
        task: str,
        completed_tasks: list,
        max_retries: int,
    ) -> dict | None:
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=self._render_state(state, task, completed_tasks)),
        ]
        for attempt in range(1, max_retries + 1):
            try:
                raw = self.llm(messages).content
                print(f"\033[36m[LLMWorker] Raw response (attempt {attempt}):\n{raw}\033[0m")
                return fix_and_parse_json(raw)
            except Exception as e:
                print(f"\033[33m[LLMWorker] Parse error (attempt {attempt}/{max_retries}): {e}\033[0m")
        return None

    def _extract_state(self, events: list) -> dict | None:
        """Pull the last observe event into a flat state dict."""
        for event_type, event in reversed(events):
            if event_type != "observe":
                continue
            status = event.get("status", {})
            pos = status.get("position", {})
            return {
                "biome": status.get("biome", "unknown"),
                "time": status.get("timeOfDay", "day"),
                "health": status.get("health", 20.0),
                "hunger": status.get("food", 20.0),
                "position": f"x={pos.get('x', 0):.1f}, y={pos.get('y', 0):.1f}, z={pos.get('z', 0):.1f}",
                "voxels": event.get("voxels", []),
                "entities": status.get("entities", {}),
                "inventory": event.get("inventory", {}),
                "equipment": status.get("equipment", []),
            }
        return None

    def _render_state(self, state: dict, task: str, completed_tasks: list) -> str:
        voxels_str = ", ".join(state["voxels"]) if state["voxels"] else "None"
        entities_str = (
            ", ".join(f"{name} ({dist:.1f}m)" for name, dist in state["entities"].items())
            if state["entities"] else "None"
        )
        inventory_str = str(state["inventory"]) if state["inventory"] else "Empty"
        completed_str = ", ".join(completed_tasks) if completed_tasks else "None"

        return (
            f"Biome: {state['biome']}\n"
            f"Time: {state['time']}\n"
            f"Health: {state['health']:.1f}/20\n"
            f"Hunger: {state['hunger']:.1f}/20\n"
            f"Position: {state['position']}\n"
            f"Nearby blocks: {voxels_str}\n"
            f"Nearby entities: {entities_str}\n"
            f"Inventory: {inventory_str}\n"
            f"Equipment: {state['equipment']}\n"
            f"\nCurrent task: {task}\n"
            f"Completed tasks so far: {completed_str}"
        )
