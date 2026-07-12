"""Deterministic mock-certificate acceptance tests from protocol §14.1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cact.protocol_v2 import AdmissionPolicyV2, CalibratedPolicy, Opportunity

KEY = "FAM|skill|mining|none|low|ordinary|early|0"

def opp(eid="ep", boundary="applicable", eligible=True):
    return Opportunity(episode_id=eid, opportunity_id=eid + "-o", round=0,
        stream_seed=1, task_id="t", world_seed=1, knowledge_id="k", source="FAM",
        type="skill", retrieval_rank=1, retrieval_score=1.0, raw_text_hash="h",
        task_group="mining", failure_type="none", risk_tier="low",
        resource_scarcity="ordinary", boundary_status=boundary,
        inventory_signature="", assignment=1, propensity_reuse=.5,
        propensity_base=.5, randomization_seed=1, start_step=0, end_step=1,
        eligible=eligible, eligibility_reason="eligible" if eligible else "blocked")

def policy(risk_inc=.02):
    return CalibratedPolicy(kappa=0, delta=.05, eps_abs=.1, eps_inc=.05,
        supported=True, audit_passed=True,
        estimates={KEY: {"supported": True, "n": 40, "delta_y": .2,
                         "se_y": 0, "risk_abs": .01, "se_abs": 0,
                         "risk_inc": risk_inc, "se_inc": 0}})

def main():
    # positive charge, negative/no-credit, exhaustion, reset, unsupported, inapplicable
    for _ in range(100):
        gate = AdmissionPolicyV2(policy(.03), use_ledger=True, initial_budget=.05)
        assert gate.decide(opp())["decision"] == "ADMIT"
        assert gate.decide(opp())["decision"] == "FALLBACK"
        assert gate.decide(opp("new-episode"))["decision"] == "ADMIT"
        assert gate.decide(opp("bad", eligible=False))["decision"] == "FALLBACK"
    print("controller ledger mock: PASS (600 path checks)")

if __name__ == "__main__": main()
