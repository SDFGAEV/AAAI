"""Smoke tests for the protocol-v2 core; no Minecraft or scipy required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
from cact.protocol_v2 import (AIPWEstimator, AdmissionPolicyV2, ApplicabilitySpec,
                              MAIN_METHODS, Opportunity, OpportunityLogger,
                              PolicyCalibrator, RandomizedAssignment)

def make_rows():
    rows = []
    assigner = RandomizedAssignment(seed=7)
    for i in range(30):
        aid, p, rseed = assigner.assign(f"opp-{i}")
        rows.append(Opportunity(
            episode_id=f"ep-{i}", opportunity_id=f"opp-{i}", round=0,
            stream_seed=1, task_id=f"task-{i%4}", world_seed=i,
            knowledge_id="k1", source="FAM", type="action_remedy",
            retrieval_rank=1, retrieval_score=0.8, raw_text_hash="h",
            task_group="mining", failure_type="wrong_tool", risk_tier="low",
            resource_scarcity="ordinary", boundary_status="applicable",
            inventory_signature="iron", assignment=aid,
            propensity_reuse=p, propensity_base=1-p,
            randomization_seed=rseed, start_step=0, end_step=10,
            y=int(aid == 1), h1=0, h2=0, h3=0, h4=0))
    return rows

def main():
    with tempfile.TemporaryDirectory() as tmp:
        logger = OpportunityLogger(Path(tmp) / "opportunities.jsonl")
        rows = make_rows()
        for row in rows: logger.append(row)
        loaded = logger.load(eligible_only=True)
        assert len(loaded) == len(rows)
        first = loaded[0].to_dict()
        for field in ("episode_phase", "prior_admission_bin", "prior_fallback_bin",
                      "prior_harm_flag", "remaining_critical_resource_ratio",
                      "time_since_last_window", "collection_exposure_count",
                      "window_type", "annotator_status", "snapshot_hash"):
            assert field in first
        estimator = AIPWEstimator(n_folds=5)
        policy = PolicyCalibrator().select(estimator.aggregate(estimator.cross_fit(loaded)))
        assert policy.kappa >= 0
        spec = ApplicabilitySpec(knowledge_id="k1", scope={"task_group": "mining"},
                                 preconditions=["has_iron_pickaxe"],
                                 hard_non_applicable=["low_health"])
        assert spec.evaluate({"has_iron_pickaxe": True}, {"task_group": "mining"})[0]
        assert not spec.evaluate({"has_iron_pickaxe": True, "low_health": True},
                                 {"task_group": "mining"})[0]
        assert AdmissionPolicyV2(policy).decide(loaded[0])["decision"] in {"ADMIT", "FALLBACK"}
        assert MAIN_METHODS == ("NoKnowledge", "NoGate", "FixedBayes",
                                "PairwisePreferenceGate", "C-ACT-Pointwise", "C-ACT")
        policy.estimates = {
            "FAM|action_remedy|mining|wrong_tool|low|ordinary|applicable": {
                "supported": True, "n": 40, "delta_y": 0.20, "se_y": 0.0,
                "risk_abs": 0.01, "se_abs": 0.0, "risk_inc": 0.03, "se_inc": 0.0,
            }
        }
        policy.delta = 0.05
        policy.eps_abs = 0.10
        policy.eps_inc = 0.05
        loaded[1].episode_id = loaded[0].episode_id
        full = AdmissionPolicyV2(policy, use_ledger=True, initial_budget=0.05)
        assert full.decide(loaded[0])["decision"] == "ADMIT"
        assert full.decide(loaded[1])["decision"] == "FALLBACK"
        pointwise = AdmissionPolicyV2(policy, use_ledger=False)
        assert pointwise.decide(loaded[0])["decision"] == "ADMIT"
        assert pointwise.decide(loaded[1])["decision"] == "ADMIT"
    print("protocol_v2 smoke: PASS")

def test_protocol_v2_smoke():
    main()

if __name__ == "__main__":
    main()

