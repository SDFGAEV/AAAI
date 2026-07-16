"""Portable tests for the dependency-free Zarr joint evidence store."""
from pathlib import Path
import tempfile, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cact.joint_evidence import JointEvidenceDrawStore
from cact.protocol_v2 import GroupEstimate

def main():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "joint_evidence_draws.zarr"
        n = 64
        estimate = GroupEstimate(
            "g0", "group", 40, 20, 20, 24.0, .2, .01, .01, .01, .01, .01, True,
            joint_draws={"delta_y": [.2] * n, "risk_inc": [.01] * n, "risk_abs": [.01] * n})
        meta = JointEvidenceDrawStore.write([estimate], path, upstream_hash="fixture", seed=17)
        assert meta["groups"] == 1 and meta["draws"] == n
        assert JointEvidenceDrawStore.validate(path)["draws"] == n
        loaded = JointEvidenceDrawStore.read(path)
        assert loaded["group"]["delta_y"] == [.2] * n
    print("joint evidence zarr: PASS")

if __name__ == "__main__":
    main()
