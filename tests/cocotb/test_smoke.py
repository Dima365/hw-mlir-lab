"""Generic contract-smoke harness (no MLIR, no socket).

Drives the IP's DUT directly through its driver using the driver's smoke_cases()
and checks the output against the Python golden baked into those cases. This
validates the driver<->IP seam (ports, handshake, packing) in isolation, before
any MLIR or runtime is involved.

Run with: make -C tests/cocotb IP=<name> MODULE=test_smoke
"""
import os

import cocotb
from cocotb.clock import Clock

from drivers import registry

IP = os.environ.get("IP", "matmul")


@cocotb.test()
async def test_smoke(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    driver = registry.BY_NAME[IP]
    await driver.setup(dut)

    cases = driver.smoke_cases()
    if not cases:
        dut._log.info("driver '%s' has no smoke_cases(); nothing to check", IP)
        return

    for i, (params, data_in, expected) in enumerate(cases):
        result = await driver.handle(dut, params, data_in)
        assert result == expected, (
            f"smoke case {i}: got {result!r}, expected {expected!r}"
        )
