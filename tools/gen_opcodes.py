#!/usr/bin/env python3
"""Generate interface/accel_opcodes.h from ips.yaml (single source of truth)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ips.yaml"
HEADER = ROOT / "interface" / "accel_opcodes.h"


def main() -> None:
    ips = yaml.safe_load(MANIFEST.read_text())
    lines = [
        "// AUTO-GENERATED from ips.yaml by tools/gen_opcodes.py. Do not edit.",
        "#ifndef ACCEL_OPCODES_H",
        "#define ACCEL_OPCODES_H",
        "",
    ]
    for ip in ips:
        lines.append(f"#define OP_{ip['name'].upper()} {ip['opcode']}")
    lines += ["", "#endif  // ACCEL_OPCODES_H", ""]
    HEADER.write_text("\n".join(lines))
    print(f"wrote {HEADER}")


if __name__ == "__main__":
    main()
