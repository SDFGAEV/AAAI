from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import OpenAI

from .goal_graph import GoalGraphManager
from .schemas import GoalGraph
from .snapshot_store import SnapshotStore


class GoalGraphLLMWorker:
    """Low-frequency worker that turns a task into a validated goal DAG.

    The worker may call an OpenAI-compatible API, but the Controller never does.
    If the API is unavailable, callers can fall back to GoalGraphManager.parse_task.
    """

    def __init__(
        self,
        goal_manager: GoalGraphManager | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        snapshot_dir: str | Path = "ckpt_evaluative/goal_graphs",
        snapshot_store: SnapshotStore | None = None,
    ):
        self.goal_manager = goal_manager or GoalGraphManager()
        self.model = model or os.environ.get("VOYAGER_DAG_MODEL") or os.environ.get(
            "VOYAGER_MODEL", "gpt-4.1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.snapshot_store = snapshot_store or SnapshotStore(
            goal_manager=self.goal_manager,
            goal_graph_dir=snapshot_dir,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate_goal_graph(self, task: str) -> GoalGraph:
        if not self.configured:
            raise RuntimeError("OPENAI_API_KEY is not configured for LLM DAG generation.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt()},
                {"role": "user", "content": f"Task: {task}"},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        data = self._parse_json(content)
        return self.goal_manager.from_dict(task, data)

    def generate_or_fallback(self, task: str) -> GoalGraph:
        graph, _, _ = self.generate_or_fallback_with_source(task)
        return graph

    def generate_or_fallback_with_source(
        self,
        task: str,
    ) -> tuple[GoalGraph, str, str | None]:
        try:
            graph = self.generate_goal_graph(task)
            print("LLM Worker generated and validated goal graph.")
            return graph, "llm", None
        except Exception as exc:
            print(f"LLM Worker fallback to local parser: {exc}")
            return self.goal_manager.parse_task(task), "fallback", str(exc)

    def save_snapshot(
        self,
        *,
        task: str,
        graph: GoalGraph,
        source: str,
        error: str | None = None,
    ) -> Path:
        return self.snapshot_store.save_goal_graph(
            task=task,
            graph=graph,
            source=source,
            model=self.model,
            base_url=self.base_url,
            error=error,
        )

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).with_name("prompts") / "goal_graph.txt"
        return prompt_path.read_text(encoding="utf-8")

    def _parse_json(self, content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        return json.loads(content)
