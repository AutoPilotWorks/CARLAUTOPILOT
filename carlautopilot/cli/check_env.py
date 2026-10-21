from __future__ import annotations

import argparse
import importlib
import shutil
import socket
import sys

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def check_python() -> tuple[str, str, str]:
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 7) <= (v.major, v.minor) <= (3, 12):
        return OK, "python", s
    return WARN, "python", f"{s} (CARLA 0.9.15 Python API supports 3.7-3.12)"


def check_module(name: str, required: bool = True) -> tuple[str, str, str]:
    try:
        m = importlib.import_module(name)
    except Exception as e:
        return (FAIL if required else WARN), name, f"import failed: {e}"
    return OK, name, str(getattr(m, "__version__", "installed"))


def check_cuda() -> tuple[str, str, str]:
    try:
        import torch
    except Exception as e:
        return WARN, "cuda", f"torch missing: {e}"
    if not torch.cuda.is_available():
        return WARN, "cuda", "torch.cuda.is_available() is False"
    p = torch.cuda.get_device_properties(0)
    return OK, "cuda", f"{p.name}, {p.total_memory / 2**30:.1f} GiB, torch cuda {torch.version.cuda}"


def check_disk(path: str = ".", need_gib: float = 20.0) -> tuple[str, str, str]:
    free = shutil.disk_usage(path).free / 2**30
    return (OK if free >= need_gib else WARN), "disk", f"{free:.0f} GiB free (CARLA needs about {need_gib:.0f})"


def check_server(host: str, port: int, timeout: float = 2.0) -> tuple[str, str, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return OK, "carla-server", f"{host}:{port} reachable"
    except OSError as e:
        return WARN, "carla-server", f"{host}:{port} not reachable ({e})"


def run_checks(host: str = "localhost", port: int = 2000, disk_path: str = ".") -> list[tuple[str, str, str]]:
    return [
        check_python(),
        check_module("numpy"),
        check_module("PIL"),
        check_module("pytest"),
        check_module("torch"),
        check_cuda(),
        check_module("carla", required=False),
        check_disk(disk_path),
        check_server(host, port),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Report-only environment check; installs nothing")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--disk-path", default=".")
    a = p.parse_args(argv)
    results = run_checks(a.host, a.port, a.disk_path)
    for status, name, detail in results:
        print(f"[{status:4s}] {name}: {detail}")
    return 1 if any(r[0] == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
