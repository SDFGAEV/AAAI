from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from .goal_graph import GoalGraphManager
from .config import ConstraintSet, ValueMatrixConfig
from .llm_worker import GoalGraphLLMWorker
from .shared_state import SharedState
from .snapshot_store import SnapshotStore


@dataclass(frozen=True)
class GoalGraphRequest:
    task: str
    use_llm: bool


class AsyncLLMWorker:
    """Background worker that updates SharedState with validated goal graphs."""

    def __init__(
        self,
        shared_state: SharedState,
        goal_manager: GoalGraphManager | None = None,
        llm_worker: GoalGraphLLMWorker | None = None,
        snapshot_store: SnapshotStore | None = None,
        value_matrix: ValueMatrixConfig | None = None,
        constraints: ConstraintSet | None = None,
    ):
        self.shared_state = shared_state
        self.goal_manager = goal_manager or GoalGraphManager()
        self.llm_worker = llm_worker or GoalGraphLLMWorker(self.goal_manager)
        self.snapshot_store = snapshot_store or SnapshotStore(
            goal_manager=self.goal_manager
        )
        self.value_matrix = value_matrix or ValueMatrixConfig()
        self.constraints = constraints or ConstraintSet()
        self._requests: queue.Queue[GoalGraphRequest | None] = queue.Queue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._last_snapshot_path: str | None = None
        self._last_shared_snapshot_path: str | None = None
        self._last_source: str | None = None
        self._last_generation_duration_ms: float = 0.0

    @property
    def last_metadata(self) -> dict:
        return {
            "source": self._last_source,
            "error": self._last_error,
            "snapshot_path": self._last_snapshot_path,
            "shared_snapshot_path": self._last_shared_snapshot_path,
            "generation_duration_ms": self._last_generation_duration_ms,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="evaluative-llm-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._requests.put(None)
        if self._thread:
            self._thread.join(timeout=timeout)

    def request_goal_graph(self, task: str, *, use_llm: bool) -> None:
        self._ready.clear()
        self._requests.put(GoalGraphRequest(task=task, use_llm=use_llm))

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout=timeout)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            self._handle_goal_graph_request(request)

    def _handle_goal_graph_request(self, request: GoalGraphRequest) -> None:
        started = time.perf_counter()
        error = None
        if request.use_llm:
            graph, source, error = self.llm_worker.generate_or_fallback_with_source(
                request.task
            )
        else:
            graph = self.goal_manager.parse_task(request.task)
            source = "fallback"

        snapshot_path = self.llm_worker.save_snapshot(
            task=request.task,
            graph=graph,
            source=source,
            error=error,
        )
        shared_snapshot = self.shared_state.update(
            goal_graph=graph,
            value_matrix=self.value_matrix.to_dict(),
            constraints=self.constraints.to_dict(),
            source=source,
        )
        shared_snapshot_path = self.snapshot_store.save_shared_snapshot(
            shared_snapshot
        )
        self._last_source = source
        self._last_error = error
        self._last_snapshot_path = str(snapshot_path)
        self._last_shared_snapshot_path = str(shared_snapshot_path)
        self._last_generation_duration_ms = round(
            (time.perf_counter() - started) * 1000,
            3,
        )
        self._ready.set()
