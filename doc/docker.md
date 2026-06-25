# Docker Environment

Docker is the primary build and test environment for this project. The host is
used for editing files and opening waveform dumps. Build, demo, and test
commands should run through Docker so that LLVM/MLIR, Clang, Verilator, cocotb,
and `standalone-opt` stay version-compatible.

## Toolchain Layout

The image builds LLVM from a pinned `llvm-project` commit and installs it into:

```text
/opt/llvm
```

The installed toolchain provides:

```text
/opt/llvm/bin/clang
/opt/llvm/bin/llc
/opt/llvm/bin/mlir-opt
/opt/llvm/bin/mlir-translate
/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake
/opt/llvm/lib/cmake/mlir/MLIRConfig.cmake
```

Cocotb is installed into:

```text
/opt/venv
```

The project does not use a host-local `.venv`. Python packages used by the
build and tests live inside the Docker image.

The project is mounted into the container as:

```text
/work
```

## Build The Image

```bash
make docker-build
```

To override the LLVM commit:

```bash
LLVM_COMMIT=<llvm-project-commit> make docker-build
```

## Build standalone-opt

`standalone-opt` must be rebuilt after changes to C++ passes, dialect
definitions, operation definitions, or standalone CMake files.

Configure and build it with:

```bash
make standalone
```

The script configures `standalone/build` if needed and then runs the build
against `/opt/llvm`.

After changing only MLIR input files, runtime C code, Python tests, or RTL, this
step is not needed.

## Run The Demo

```bash
make demo
```

This runs the MLIR pipeline, compiles the generated LLVM IR, and executes the
result through the cocotb/Verilator bridge.

## Run Integration Tests

```bash
make test
```

## Open A Shell

```bash
make shell
```

## Open Waveforms

The cocotb pipeline writes an FST waveform:

```text
tests/cocotb/dump.fst
```

Open it on the host:

```bash
surfer tests/cocotb/dump.fst
```

The wave viewer is intentionally not part of the Docker image.

## Clean Generated Files

Generated MLIR/test artifacts and cocotb simulation outputs:

```bash
make clean-generated
```

The Docker-owned standalone build directory:

```bash
make clean-standalone
```

Use `clean-standalone` when `standalone/build/CMakeCache.txt` was created by a
host build or another incompatible path. The next `make standalone` will
configure it again inside Docker.
