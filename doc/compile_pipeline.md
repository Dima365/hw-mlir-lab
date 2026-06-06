# Compile Pipeline

This file describes the steps in `pipelines/compile_pipeline.sh`.

`pipelines/compile_pipeline.sh` builds an executable from the MLIR pipeline
output and a C driver. It is used by `demo/run.sh` and the tests in
`tests/matmul/build.py`.

Interface:

```bash
./pipelines/compile_pipeline.sh <llvm-ir> <main.c> <app> <object-dir>
```

## 1. Interface Object

Input: `interface/interface.c`

Output: `<object-dir>/interface.o`

The interface bridge is compiled into an object file. It provides the integer
implementation of `systolic_matmul_8x8`, which connects to the Python/Verilator
simulator through a Unix socket. The runtime ABI uses `i8` inputs and an `i32`
accumulator.

## 2. MLIR Program Object

Input: `<llvm-ir>`

Output: `<object-dir>/mlir_program.o`

`llc` compiles the LLVM IR produced from MLIR into an object file.

## 3. Link App

Inputs: `<object-dir>/mlir_program.o`, `<object-dir>/interface.o`, `<main.c>`

Output: `<app>`

`clang` links the MLIR object, interface object, C driver, and MLIR runner utils
into the final executable. Linking uses the LLVM linker `ld.lld` from
`/opt/llvm/bin`.
