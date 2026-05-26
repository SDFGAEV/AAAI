"""
A2 Sanity Run: Craft 1 stone pickaxe
Shadow mode: Voyager handles actual execution; LLMWorker + Controller run in parallel
and log what the evaluative-interface system would decide at each step.
"""
import copy
import json
import os
import sys

# Must be before any other imports to bypass macOS proxy for localhost
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

keys_path = os.path.join(os.path.dirname(__file__), "../MC复现/mindcraft-develop/keys.json")
with open(keys_path) as f:
    keys = json.load(f)
os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]

from voyager import Voyager
from voyager.agents import (
    ValueMatrix, ConstraintEngine, EntityRegistry,
    LLMWorker, Controller,
)

# ─── Config ───────────────────────────────────────────────────────────────────
TASK = "Craft 1 stone pickaxe"
MC_PORT = 55916
CKPT = os.path.join(os.path.dirname(__file__), "ckpt/a2_sanity_run4")
MAX_RETRIES = 4   # per task (Voyager default)

# ─── Init Voyager ─────────────────────────────────────────────────────────────
voyager = Voyager(
    mc_port=MC_PORT,
    openai_api_key=keys["OPENAI_API_KEY"],
    action_agent_model_name="gpt-4o",
    curriculum_agent_model_name="gpt-4o",
    curriculum_agent_qa_model_name="gpt-4o-mini",
    critic_agent_model_name="gpt-4o",
    skill_manager_model_name="gpt-4o-mini",
    max_iterations=MAX_RETRIES + 2,
    openai_api_request_timeout=60,
    ckpt_dir=CKPT,
)

# ─── Init evaluative-interface components ─────────────────────────────────────
vm = ValueMatrix()
ce = ConstraintEngine()
er = EntityRegistry()
worker = LLMWorker(model_name="gpt-4o", temperature=0, request_timeout=60)
ctrl = Controller(vm, ce, er)

# ─── Hard reset: clear inventory, get initial events ─────────────────────────
print("\n\033[35m[A2 Sanity] Hard-resetting environment...\033[0m")
initial_events = voyager.env.reset(options={"mode": "hard", "wait_ticks": 20})
voyager.resume = True   # don't hard-reset again in learn()

# ─── LLMWorker: generate V / C / G for the target task ───────────────────────
print(f"\n\033[35m[A2 Sanity] Calling LLMWorker for task: {TASK!r}\033[0m")
dag = worker.update_from_events(
    events=initial_events,
    task=TASK,
    completed_tasks=[],
    value_matrix=vm,
    constraint_engine=ce,
    entity_registry=er,
)
ctrl.set_dag(dag)

print(f"\n\033[35m[A2 Sanity] TaskDAG: {dag}\033[0m")
if dag:
    leaves = dag.executable_leaves()
    print(f"\033[35m[A2 Sanity] Executable leaves: {[(n.action_type, n.target) for n in leaves]}\033[0m")

# ─── Controller shadow: initial recommendation ────────────────────────────────
print(f"\n\033[34m[Controller shadow] Initial recommendation:\033[0m")
ctrl.select_action(initial_events)

# ─── Custom rollout loop with Controller shadow ───────────────────────────────
print(f"\n\033[35m[A2 Sanity] Starting task: {TASK!r} (max {MAX_RETRIES} retries)\033[0m")

shadow_log = []   # list of (step, voyager_action_summary, controller_action)

# Reset Voyager agent state for this task (env already reset above)
voyager.action_agent_rollout_num_iter = 0
voyager.task = TASK
context = voyager.curriculum_agent.get_task_context(TASK)
voyager.context = context

# Peek observation (needed by reset to build messages); reuse initial_events
events = initial_events
difficulty = "peaceful"
peek_events = voyager.env.step(
    "bot.chat(`/time set ${getNextTime()}`);\n"
    + f"bot.chat('/difficulty {difficulty}');"
)
skills = voyager.skill_manager.retrieve_skills(query=context)
system_message = voyager.action_agent.render_system_message(skills=skills)
human_message = voyager.action_agent.render_human_message(
    events=peek_events, code="", task=TASK, context=context, critique=""
)
voyager.messages = [system_message, human_message]
voyager.conversations = []
voyager.last_events = copy.deepcopy(peek_events)

step_num = 0
success = False
while step_num < MAX_RETRIES:
    step_num += 1
    print(f"\n\033[35m[A2 Sanity] ── Step {step_num}/{MAX_RETRIES} ──\033[0m")

    messages, reward, done, info = voyager.step()
    success = info["success"]

    # Shadow: what would Controller choose after seeing these events?
    if voyager.last_events:
        print(f"\n\033[34m[Controller shadow step {step_num}] Recommendation:\033[0m")
        ctrl_action = ctrl.select_action(voyager.last_events)
        shadow_log.append({
            "step": step_num,
            "controller_action": f"{ctrl_action.action_type}:{ctrl_action.target}",
            "controller_score": round(ctrl_action.final_score, 3),
            "goal_bonus": ctrl_action.goal_bonus,
        })
        # Advance DAG if action matches
        ctrl.mark_action_done(ctrl_action)

    if done:
        break

# ─── Result ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"[A2 Sanity] Task: {TASK!r}")
print(f"[A2 Sanity] Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
print(f"[A2 Sanity] Steps taken: {step_num}")
print(f"\n[A2 Sanity] Controller shadow log:")
for entry in shadow_log:
    bonus_str = f"  (+goal_bonus)" if entry["goal_bonus"] > 0 else ""
    print(f"  step {entry['step']}: {entry['controller_action']}  score={entry['controller_score']}{bonus_str}")

# Save shadow log
os.makedirs(CKPT, exist_ok=True)
log_path = os.path.join(CKPT, "shadow_log.json")
with open(log_path, "w") as f:
    json.dump({
        "task": TASK,
        "success": success,
        "steps": step_num,
        "shadow_log": shadow_log,
        "dag": dag.to_dict() if dag else None,
    }, f, indent=2)
print(f"\n[A2 Sanity] Shadow log saved to {log_path}")

voyager.env.close()
