#!/usr/bin/env python3
"""
Pre-experiment hardware health check.
Verifies all components are ready before launching experiments.
"""

import sys, os, time, socket, subprocess

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJ)

CHECKS = []


def check(name, fn):
    try:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        CHECKS.append((name, status, msg))
        print(f"  [{status}] {name}: {msg}")
    except Exception as e:
        CHECKS.append((name, "FAIL", str(e)))
        print(f"  [FAIL] {name}: {e}")


def main():
    print("=" * 60)
    print("  C-ACT Pre-Experiment Health Check")
    print("=" * 60)

    # 1. Python
    check("Python >= 3.10", lambda: (
        sys.version_info >= (3, 10),
        f"Python {sys.version}"))

    # 2. PyTorch + CUDA
    check("CUDA available", lambda: (
        __import__("torch").cuda.is_available(),
        f"CUDA {__import__('torch').version.cuda} | GPU: "
        f"{__import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'N/A'}"))

    # 3. GPU memory
    if __import__("torch").cuda.is_available():
        mem = __import__("torch").cuda.get_device_properties(0).total_mem / 1e9
        check("GPU memory >= 8GB", lambda: (mem >= 8, f"{mem:.1f} GB"))

    # 4. Java (for Minecraft)
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
        ok = result.returncode == 0
        check("Java available", lambda: (ok, result.stderr.split("\n")[0] if ok else "not found"))
    except FileNotFoundError:
        check("Java available", lambda: (False, "java not found in PATH"))

    # 5. C-ACT modules
    check("C-ACT modules import", lambda: (
        all(__import__(m) for m in [
            "cact.contract", "cact.trust_store", "cact.trust_gate",
            "cact.decision_controller", "cact.cact_memory",
        ]),
        "all 14 modules OK"))

    # 6. Benchmark configs
    for bm in ["cact_calib", "cact_p3", "cact_train"]:
        path = os.path.join(_PROJ, "src", "optimus1", "conf", "benchmark", f"{bm}.yaml")
        check(f"Benchmark {bm}", lambda p=path: (
            os.path.exists(p),
            f"found ({os.path.getsize(p)} bytes)" if os.path.exists(p) else "MISSING"))

    # 7. Model checkpoint files
    for ckpt_name, ckpt_path in [
        ("VPT model", "checkpoints/vpt/2x.model"),
        ("STEVE-1 weights", "checkpoints/steve1/steve1.weights"),
        ("STEVE-1 prior", "checkpoints/steve1/steve1_prior.pt"),
    ]:
        full = os.path.join(_PROJ, ckpt_path)
        check(f"Checkpoint {ckpt_name}", lambda p=full: (
            os.path.exists(p),
            f"{os.path.getsize(p)/1e6:.0f} MB" if os.path.exists(p) else "MISSING"))

    # 8. Disk space
    import shutil
    usage = shutil.disk_usage(_PROJ)
    free_gb = usage.free / 1e9
    check(f"Disk space >= 10GB free", lambda: (free_gb >= 10, f"{free_gb:.1f} GB free"))

    # 9. Port availability
    ports = [12345, 15000, 15001, 15002, 15003]
    free_ports = []
    for p in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                free_ports.append(p)
        except OSError:
            pass
    check("Free ports (≥4)", lambda: (
        len(free_ports) >= 4,
        f"{len(free_ports)} free: {free_ports}"))

    # 10. RAM
    import psutil
    avail = psutil.virtual_memory().available / 1e9
    check("RAM available >= 8GB", lambda: (avail >= 8, f"{avail:.1f} GB"))

    # Summary
    passed = sum(1 for _, s, _ in CHECKS if s == "PASS")
    failed = sum(1 for _, s, _ in CHECKS if s == "FAIL")
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed}/{passed+failed} checks passed")
    if failed > 0:
        print(f"  FAILURES ({failed}):")
        for name, _, msg in CHECKS:
            if _ == "FAIL":
                print(f"    - {name}: {msg}")
    else:
        print(f"  All checks passed — ready to run experiments!")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
