# hw-mlir-lab

Experimental MLIR-to-systolic-array compilation and RTL simulation flow.

This project explores how MLIR can be used as a compiler layer between high-level linear algebra operations and custom accelerator IP blocks.

The current MVP provides an end-to-end flow for a fixed-size 8x8 systolic array: it maps `linalg.matmul` operations to the accelerator interface and validates execution against a cocotb/Verilator simulation of a SystemVerilog RTL block.

While the current implementation focuses on a single systolic-array accelerator, the broader goal is to evolve this project into a framework for accelerator subsystem prototyping and experimentation. The long-term vision is to use MLIR as a central integration layer for combining compiler transformations, accelerator IP blocks, runtime interfaces, and RTL-based validation within a reproducible workflow. Future directions may include support for additional accelerator types, tighter integration with CIRCT, and synthesis-driven evaluation to help relate compiler decisions to hardware implementation metrics such as area, timing, and resource utilization.

## What This Project Does

The current flow takes a high-level MLIR matrix multiplication and lowers selected operations to calls into a simulated hardware accelerator.

```text
linalg.matmul
  -> tiling / padding
  -> standalone.systolic_matmul
  -> C ABI call
  -> C bridge
  -> cocotb / Verilator
  -> SystemVerilog systolic array RTL
```

The project demonstrates an end-to-end compiler/runtime/simulation path:

* transform `linalg.matmul` into hardware-sized tiles;
* pad boundary tiles to the fixed 8x8 hardware shape;
* replace suitable matmul operations with a custom MLIR operation;
* lower that operation to a C-callable function;
* execute the generated program against an RTL simulation;
* compare the result with expected output.

## Project Status

This is an experimental MVP.

Currently supported:

* `linalg.matmul` input;
* 8x8 systolic-array tile shape;
* `i8` inputs;
* `i32` accumulator/result;
* custom `standalone.systolic_matmul` operation;
* lowering to `systolic_matmul_8x8`;
* C bridge between generated code and cocotb;
* Unix socket transport for simulation;
* Verilator + cocotb RTL simulation;
* generated matmul tests.

Not supported yet:

* automatic RTL generation from MLIR (CIRCT);
* full memory hierarchy;
* DMA/interconnect modeling;
* cost model;
* automatic CPU-vs-accelerator placement;
* FPGA deployment flow;
* synthesis/timing/resource reporting.

## Architecture

```text
MLIR input
  |
  v
MLIR transform pipeline
  |
  v
custom standalone dialect / passes
  |
  v
LLVM lowering
  |
  v
native executable
  |
  v
C interface bridge
  |
  v
cocotb / Verilator
  |
  v
SystemVerilog systolic array RTL
```

The project has three important boundaries:

```text
MLIR level   - represents and transforms the computation
C ABI level  - connects generated code to the accelerator call
RTL level    - executes the operation in hardware simulation
```

## Repository Layout

```text
demo/
  Example MLIR input and C driver.

transforms/
  MLIR Transform dialect schedules for tiling and padding.

standalone/
  Custom MLIR dialect, operations, passes, and standalone-opt tool.

pipelines/
  Shell scripts for MLIR lowering, compilation, and cocotb execution.

interface/
  C bridge between generated code and the simulated hardware accelerator.

ip/
  SystemVerilog RTL IP blocks.

tests/
  Generated matmul tests and cocotb/Verilator testbench.

doc/
  Additional project documentation.
```

## Main Components

### MLIR Input

The demo starts from an MLIR program containing `linalg.matmul`.

Example location:

```text
demo/matmul.mlir
```

### Transform Pipeline

The transform pipeline prepares matmul operations for the hardware target.

Current transform schedules:

```text
transforms/matmul_tile.mlir
transforms/matmul_pad.mlir
```

The tiling step splits matmul into 8x8x8 tiles.

The padding step pads boundary tiles so that the systolic array always receives fixed-size inputs.

### Custom MLIR Operation

The project defines a custom operation:

```text
standalone.systolic_matmul
```

This operation marks a matmul tile that should be executed through the systolic array accelerator path.

### standalone-opt

The project uses `standalone-opt`, based on the MLIR standalone example.

It is an `opt`-like MLIR tool extended with project-specific dialects, operations, and passes.

In this project it is used to:

* run transform schedules;
* bufferize tensor-level IR to memref-level IR;
* create C interface wrappers;
* convert suitable `linalg.matmul` operations to `standalone.systolic_matmul`;
* lower `standalone.systolic_matmul` to a regular function call.

### C Interface Bridge

The custom MLIR operation is lowered to:

```c
systolic_matmul_8x8(...)
```

This function is implemented in:

```text
interface/interface.c
```

It sends input tiles and accumulator data to the cocotb testbench through a Unix socket.

Current ABI shape:

```text
A: i8[8][8]
B: i8[8][8]
C: i32[8][8]
```

The result is returned as an updated `i32[8][8]` accumulator.

### RTL Simulation

The RTL systolic array is implemented in SystemVerilog.

Example location:

```text
ip/systolic_array_demo/
```

The cocotb testbench receives requests from the C bridge, drives the RTL inputs, waits for completion, and returns the result back to the generated executable.

## Quick Start

The project is Docker-first.

LLVM/MLIR, Clang, Verilator, cocotb, and the custom `standalone-opt` tool are expected to run inside the Docker environment.

### Build Docker Image

```bash
make docker-build
```

### Run Demo

```bash
make demo
```

This command builds the MLIR tool, runs the MLIR pipeline, compiles the generated program, links it with the C bridge, and runs it against the cocotb/Verilator RTL simulation.

### Run Tests

```bash
make test
```

The test flow generates multiple matmul cases, runs them through the same compiler/runtime/simulation path, and checks the computed results.

## MLIR Pipeline

The MLIR pipeline saves intermediate files so the lowering process can be inspected step by step.

Typical output files:

```text
build/mlir-pipeline/01_tiled.mlir
build/mlir-pipeline/02_padded.mlir
build/mlir-pipeline/03_bufferized.mlir
build/mlir-pipeline/04_entry_wrapped.mlir
build/mlir-pipeline/05_systolic_memref.mlir
build/mlir-pipeline/06_systolic_call.mlir
build/mlir-pipeline/07_llvm.mlir
build/mlir-pipeline/08_llvm.ll
```

Pipeline stages:

```text
linalg.matmul
  |
  | tiling: 8x8x8
  v
tiled linalg.matmul
  |
  | padding
  v
fixed-size matmul tiles
  |
  | bufferization
  v
memref-level IR
  |
  | convert-linalg-matmul-to-systolic
  v
standalone.systolic_matmul
  |
  | lower-systolic-to-func-call
  v
func.call @systolic_matmul_8x8
  |
  v
LLVM dialect
  |
  v
LLVM IR
```

## Compile Pipeline

The compile pipeline builds a native executable from the MLIR pipeline output.

It:

1. compiles `interface/interface.c` into an object file;
2. compiles generated LLVM IR into an object file;
3. links the MLIR object, interface object, C driver, and MLIR runner utilities.

## Cocotb / Verilator Pipeline

The cocotb pipeline runs the compiled executable together with the RTL simulation.

For each matmul request:

1. the generated program calls `systolic_matmul_8x8`;
2. the C bridge connects to the cocotb Unix socket;
3. cocotb receives `A`, `B`, and `C`;
4. cocotb drives the SystemVerilog systolic array;
5. the testbench waits for `done`;
6. the result is sent back to the executable.

## Roadmap

Possible future directions:

* add more accelerator IP blocks;
* support more data types and layouts;
* improve hardware constraints in the compiler pipeline;
* add cost model support;
* add CPU-vs-accelerator placement decisions;
* improve runtime abstraction;
* add new the socket bridge with MMIO/DMA/FPGA-oriented interfaces;
* add synthesis sanity checks;
* investigate CIRCT integration;
* add waveform/debug documentation;
* expand the test suite.

## Contributing

Contributions, issues, and architecture discussions are welcome.

Interesting areas for contribution:

* new MLIR passes;
* new accelerator IP blocks;
* cocotb/Verilator tests;
* runtime interface experiments;
* synthesis flow experiments;
* CIRCT-related experiments;
* documentation and examples.

## License

Apache License 2.0. See the LICENSE file for details.

## Related Topics

* MLIR
* LLVM
* linalg dialect
* Transform dialect
* custom MLIR dialects
* systolic arrays
* Verilator
* cocotb
* RTL simulation
* accelerator prototyping
