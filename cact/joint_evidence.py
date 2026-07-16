"""Dependency-free Zarr v2 store for paired C-ACT evidence draws.

The server may not have the optional ``zarr`` package.  This module writes
valid uncompressed Zarr v2 metadata/chunks directly, so the artifact remains
portable and inspectable by standard Zarr readers.
"""
from __future__ import annotations
import hashlib, json, math, shutil
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

SCHEMA = "cact.joint_evidence_draws.v1"

def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _zarray(path: Path, shape: Sequence[int]) -> None:
    _write_json(path / ".zarray", {
        "zarr_format": 2, "shape": list(shape), "chunks": list(shape),
        "dtype": "<f8", "compressor": None, "fill_value": 0.0,
        "order": "C", "filters": None
    })

def _safe_float_array(values: Any, key: str) -> np.ndarray:
    arr = np.asarray(values, dtype="<f8")
    if arr.ndim != 1 or arr.size < 32 or not np.all(np.isfinite(arr)):
        raise ValueError(f"joint draw array {key} must be finite 1-D with >=32 draws")
    return arr

class JointEvidenceDrawStore:
    """Write/read paired benefit, incremental-harm and absolute-risk draws."""
    @staticmethod
    def write(estimates: Sequence[Any], path: str | Path, *, upstream_hash: str = "", seed: int = 0) -> dict[str, Any]:
        out = Path(path)
        if out.exists():
            if out.is_dir(): shutil.rmtree(out)
            else: out.unlink()
        out.mkdir(parents=True)
        (out / "groups").mkdir()
        _write_json(out / ".zgroup", {"zarr_format": 2})
        keys = []
        for idx, estimate in enumerate(sorted(estimates, key=lambda x: str(x.key))):
            draws = dict(getattr(estimate, "joint_draws", {}) or {})
            y = _safe_float_array(draws.get("delta_y", []), "delta_y")
            inc = _safe_float_array(draws.get("risk_inc", []), "risk_inc")
            abs_r = _safe_float_array(draws.get("risk_abs", []), "risk_abs")
            if not (len(y) == len(inc) == len(abs_r)):
                raise ValueError(f"joint draw length mismatch for {estimate.key}")
            group = out / "groups" / f"g{idx:06d}"
            group.mkdir()
            _write_json(group / ".zgroup", {"zarr_format": 2})
            for name, arr in (("delta_y", y), ("risk_inc", inc), ("risk_abs", abs_r)):
                arr_dir = group / name; arr_dir.mkdir(); _zarray(arr_dir, arr.shape); arr.tofile(arr_dir / "0")
            keys.append({"key": str(estimate.key), "group": f"groups/g{idx:06d}", "draws": int(len(y))})
        attrs = {"schema_version": SCHEMA, "upstream_hash": str(upstream_hash),
                 "seed": int(seed), "groups": keys}
        _write_json(out / ".zattrs", attrs)
        digest = hashlib.sha256()
        for file in sorted(out.rglob("*")):
            if file.is_file():
                digest.update(file.relative_to(out).as_posix().encode()); digest.update(file.read_bytes())
        attrs["sha256"] = digest.hexdigest(); _write_json(out / ".zattrs", attrs)
        return {"path": str(out), "sha256": attrs["sha256"], "groups": len(keys), "draws": sum(x["draws"] for x in keys)}

    @staticmethod
    def read(path: str | Path) -> dict[str, dict[str, list[float]]]:
        root = Path(path); attrs = _read_json(root / ".zattrs")
        if attrs.get("schema_version") != SCHEMA:
            raise ValueError("unsupported joint evidence store schema")
        result = {}
        for entry in attrs.get("groups", []):
            group = root / entry["group"]; arrays = {}
            for name in ("delta_y", "risk_inc", "risk_abs"):
                meta = _read_json(group / name / ".zarray")
                if meta.get("dtype") != "<f8" or meta.get("zarr_format") != 2:
                    raise ValueError(f"unsupported zarr array: {name}")
                shape = tuple(meta.get("shape", [])); arr = np.fromfile(group / name / "0", dtype="<f8")
                if arr.shape != shape or not np.all(np.isfinite(arr)):
                    raise ValueError(f"corrupt joint draw array: {entry['key']}/{name}")
                arrays[name] = arr.tolist()
            result[str(entry["key"])] = arrays
        return result

    @staticmethod
    def validate(path: str | Path) -> dict[str, Any]:
        data = JointEvidenceDrawStore.read(path)
        lengths = {key: len(values["delta_y"]) for key, values in data.items()}
        if not data or min(lengths.values()) < 32 or len(set(lengths.values())) != 1:
            raise ValueError("joint evidence store has insufficient or inconsistent draws")
        return {"schema_version": SCHEMA, "groups": len(data), "draws": next(iter(lengths.values()))}
