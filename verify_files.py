#!/usr/bin/env python3
"""Server-side verification: python3 verify_files.py"""
import os

FILES = [
    "app.py", "pyproject.toml", "setup_ubuntu.sh", ".gitignore",
    "cact/__init__.py", "cact/active_logging.py", "cact/cact_memory.py",
    "cact/context_bucket.py", "cact/contract.py", "cact/decision_controller.py",
    "cact/empirical_bayes.py", "cact/interaction_gate.py", "cact/lifecycle_manager.py",
    "cact/metrics.py", "cact/temporal_decay.py", "cact/thompson_probe.py",
    "cact/trust_gate.py", "cact/trust_store.py",
    "xenon_integration/__init__.py", "xenon_integration/adg_parser.py",
    "xenon_integration/executor_wrapper.py", "xenon_integration/fam_parser.py",
    "xenon_integration/xenon_adapter.py",
    "experiments/batch_proxy.py", "experiments/health_check.py",
    "experiments/parallel_runner.py", "experiments/run_all.sh",
    "experiments/vlm_cache.py",
    "configs/env.yaml",
    "configs/methods/no_knowledge.yaml", "configs/methods/xenon_original.yaml",
    "configs/methods/bank_curation.yaml", "configs/methods/lifecycle_success.yaml",
    "configs/methods/fixed_bayes.yaml", "configs/methods/act.yaml",
    "configs/methods/cact_full.yaml",
    "tests/test_cact_stress.py", "tests/test_mathematical_correctness.py",
    "tests/test_end_to_end_scenarios.py",
    "analysis/compute_metrics.py", "analysis/plot_risk_coverage.py",
    "analysis/plot_lifecycle.py",
    "scripts/run_exploration.sh", "scripts/run_planning_diamond.sh",
    "src/_fcntl_stub.py",
    "src/optimus1/main_planning.py", "src/optimus1/main_exploration.py",
    "src/optimus1/conf/evaluate.yaml",
    "src/optimus1/conf/benchmark/cact_calib.yaml",
    "src/optimus1/conf/benchmark/cact_p3.yaml",
    "src/optimus1/conf/benchmark/cact_train.yaml",
    "src/optimus1/server/agent.py",
    "src/optimus1/server/api/request.py", "src/optimus1/server/api/utils.py",
    "src/optimus1/env/custom_env.py", "src/optimus1/env/wrapper.py",
    "src/optimus1/env/chat_action.py", "src/optimus1/env/inventory_agent_start.py",
    "src/optimus1/env/obversation_current_location.py",
    "src/optimus1/env/plain_inventory.py",
    "src/optimus1/env/mods/mod.py", "src/optimus1/env/mods/recorder.py",
    "src/optimus1/env/mods/status.py", "src/optimus1/env/mods/task_checker.py",
    "src/optimus1/helper/new_helper.py", "src/optimus1/helper/new_craft_helper.py",
    "src/optimus1/helper/new_equip_helper.py", "src/optimus1/helper/new_smelt_helper.py",
    "src/optimus1/helper/slot.py", "src/optimus1/helper/tag_items.json",
    "src/optimus1/memories/decomposed_memory.py",
    "src/optimus1/memories/relative_graph.py",
    "src/optimus1/memories/hypothesized_recipe_graph.py",
    "src/optimus1/models/base_model.py", "src/optimus1/models/qwen_vl_planning.py",
    "src/optimus1/models/steve_action_model.py", "src/optimus1/models/utils.py",
    "src/optimus1/monitor/monitor.py", "src/optimus1/monitor/monitors.py",
    "src/optimus1/monitor/step_monitor.py", "src/optimus1/monitor/success_monitor.py",
    "src/optimus1/util/decorator.py", "src/optimus1/util/image.py",
    "src/optimus1/util/logger.py", "src/optimus1/util/prompt.py",
    "src/optimus1/util/server_api.py", "src/optimus1/util/thread.py",
    "src/optimus1/util/tools.py", "src/optimus1/util/utils.py",
    "src/optimus1/util/video.py",
    "paper/main.tex", "paper/references.bib", "paper/aaai25.sty",
    "paper/aaai25.bst", "paper/AAAI_WRITING_PATTERNS.md", "paper/CITATION_AUDIT.md",
]

missing = []
for f in FILES:
    if not os.path.exists(f):
        missing.append(f)
        print(f"MISSING: {f}")

if not missing:
    print(f"OK: All {len(FILES)} files present.")
else:
    print(f"FAIL: {len(missing)}/{len(FILES)} files MISSING!")
