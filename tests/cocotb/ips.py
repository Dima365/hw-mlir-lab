"""IP manifest (ips.yaml) loader — single source of truth for IP blocks."""
from pathlib import Path

import yaml

_MANIFEST = Path(__file__).resolve().parents[2] / "ips.yaml"


def load() -> list[dict]:
    return yaml.safe_load(_MANIFEST.read_text())


def by_name(name: str) -> dict:
    for ip in load():
        if ip["name"] == name:
            return ip
    raise KeyError(f"IP '{name}' not found in {_MANIFEST}")


def by_opcode(opcode: int) -> dict:
    for ip in load():
        if ip["opcode"] == opcode:
            return ip
    raise KeyError(f"opcode {opcode} not found in {_MANIFEST}")


def gargs(name: str) -> str:
    """Verilator -G parameter flags for an IP, from its manifest params."""
    params = by_name(name).get("params", {})
    return " ".join(f"-G{key}={value}" for key, value in params.items())


def _main(argv: list[str]) -> None:
    """Tiny CLI so the Makefile can query the manifest by IP name."""
    if len(argv) != 3:
        raise SystemExit("usage: ips.py {toplevel|sources|gargs} <name>")
    cmd, name = argv[1], argv[2]
    if cmd == "toplevel":
        print(by_name(name)["module"])
    elif cmd == "sources":
        print(" ".join(by_name(name)["sources"]))
    elif cmd == "gargs":
        print(gargs(name))
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    import sys

    _main(sys.argv)
