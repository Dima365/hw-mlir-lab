"""Base class for IP drivers used by the generic cocotb harness.

One driver per IP block. A driver knows how to drive its DUT's ports and how to
pack/unpack the request payload. The generic harness (test_socket.py) selects a
driver by opcode and only moves bytes; all DUT-specific knowledge lives here.

A driver references its IP by NAME; metadata (opcode, sources, params) lives in
the manifest ips.yaml, not in the driver.
"""


class IPDriver:
    NAME: str

    async def setup(self, dut):
        """Optional per-IP init (e.g. reset). Default: no-op."""
        return

    async def handle(self, dut, params: bytes, data_in: bytes) -> bytes:
        """Drive the DUT for one request and return the output bytes.

        params   -- scalar parameters (param_bytes from the header)
        data_in  -- input data (in_bytes from the header)
        returns  -- output bytes sent back to the app (must be out_bytes long)
        """
        raise NotImplementedError
