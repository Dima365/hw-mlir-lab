"""Registry of IP drivers, keyed by manifest name and opcode.

Opcodes come from the manifest (ips.yaml) — the single source of truth — so a
driver never hardcodes its opcode. To add an IP: implement a driver and append
its instance to _DRIVERS.
"""
import ips

from .conv_requant import ConvRequantDriver
from .matmul import MatmulDriver
from .qadd_relu import QAddReluDriver

_DRIVERS = [MatmulDriver(), QAddReluDriver(), ConvRequantDriver()]

BY_NAME = {d.NAME: d for d in _DRIVERS}
BY_OPCODE = {ips.by_name(d.NAME)["opcode"]: d for d in _DRIVERS}
