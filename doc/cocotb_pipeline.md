# Cocotb Pipeline

This file describes `pipelines/cocotb_pipeline.sh`.

The script runs the compiled app together with the cocotb/Verilator testbench
for `ip/systolic_array_demo/array.sv`.

Interface:

```bash
./pipelines/cocotb_pipeline.sh <app> [ip]
```

`ip` selects which IP block to simulate (default `matmul`); its toplevel,
sources, and parameters come from the manifest `ips.yaml`.

## 1. Start Cocotb

The script invokes the generic Makefile:

```bash
make -C tests/cocotb IP=<ip>
```

The following environment variables are passed:

```text
APP       path to the compiled app
REPO_ROOT repository root
IP        IP block name (selects toplevel/sources/params from ips.yaml)
```

## 2. Socket Bridge

The cocotb testbench creates a Unix socket and a ready file:

```text
/tmp/systolic_cocotb.sock
/tmp/systolic_cocotb.ready
```

After that, cocotb starts `<app>`. The runtime function `systolic_matmul_8x8`
from `interface/interface.c` connects to this socket.

## 3. Drive RTL

For each matmul request, cocotb:

- reads the `i8` A and B matrices;
- reads the `i32` C accumulator;
- loads A, B, and C into `a_flat`, `b_flat`, and `c_in_flat`;
- asserts `start`;
- waits for `done`;
- reads `c_out_flat`;
- returns the `i32` result back to the app.
