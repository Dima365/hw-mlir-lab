"""Registry of IP drivers, keyed by manifest name and opcode.

Opcodes come from the manifest (ips.yaml) — the single source of truth — so a
driver never hardcodes its opcode. To add an IP: implement a driver and append
its instance to _DRIVERS.
"""
import ips

from .accelerator import AcceleratorDriver
from .conv_requant import ConvRequantDriver
from .matmul import MatmulDriver
from .qadd_relu import QAddReluDriver

_LEAF_DRIVERS = [MatmulDriver(), QAddReluDriver(), ConvRequantDriver()]
BY_OPCODE = {
    ips.by_name(driver.NAME)["opcode"]: driver
    for driver in _LEAF_DRIVERS
}
_ACCELERATOR = AcceleratorDriver(
    {
        opcode: (driver, driver.NAME)
        for opcode, driver in BY_OPCODE.items()
    }
)
_DRIVERS = [*_LEAF_DRIVERS, _ACCELERATOR]

BY_NAME = {d.NAME: d for d in _DRIVERS}


async def handle_request(ip_name, dut, opcode, params, data_in):
    top_driver = BY_NAME[ip_name]
    if isinstance(top_driver, AcceleratorDriver):
        return await top_driver.handle_opcode(dut, opcode, params, data_in)

    request_driver = BY_OPCODE[opcode]
    if request_driver is not top_driver:
        raise ValueError(
            f"DUT '{ip_name}' cannot execute opcode {opcode} "
            f"for '{request_driver.NAME}'"
        )
    return await request_driver.handle(dut, params, data_in)
