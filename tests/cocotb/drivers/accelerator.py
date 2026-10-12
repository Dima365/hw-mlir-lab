"""Router for the combined accelerator simulation top."""

from .base import IPDriver


class _PrefixedDut:
    def __init__(self, dut, prefix):
        self._dut = dut
        self._prefix = prefix

    def __getattr__(self, name):
        if name in ("clk", "rst"):
            return getattr(self._dut, name)
        return getattr(self._dut, f"{self._prefix}_{name}")


class AcceleratorDriver(IPDriver):
    NAME = "accelerator"

    def __init__(self, routes):
        self._routes = routes

    async def setup(self, dut):
        for driver, prefix in self._routes.values():
            await driver.setup(_PrefixedDut(dut, prefix))

    async def handle_opcode(self, dut, opcode, params, data_in):
        try:
            driver, prefix = self._routes[opcode]
        except KeyError as error:
            raise ValueError(f"accelerator has no route for opcode {opcode}") \
                from error
        return await driver.handle(
            _PrefixedDut(dut, prefix), params, data_in
        )
