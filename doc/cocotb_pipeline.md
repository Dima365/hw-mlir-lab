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

After that, cocotb starts `<app>`. The runtime functions
`systolic_matmul_8x8` and `systolic_u8s8_matmul_8x8` from
`interface/interface.c` connect to this socket.

## 3. Drive RTL

For each matmul request, cocotb:

- reads `a_is_unsigned` and `a_zero_point` command parameters;
- reads signed `i8` A or quantized `uint8` A and signed `i8` B;
- reads the `i32` C accumulator;
- loads the matrices and quantization parameters into the RTL ports;
- asserts `start`;
- waits for `done`;
- reads `c_out_flat`;
- returns the `i32` result back to the app.

The signed mode computes `C = C_in + A_i8 * B_i8`. The quantized mode computes
`C = C_in + (A_u8 - a_zero_point) * B_i8`.
