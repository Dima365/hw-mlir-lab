# MLIR Pipeline

This file describes the steps in `pipelines/mlir_pipeline.sh`. By default, the
input file is `demo/matmul.mlir`, and intermediate results are written to
`build/mlir-pipeline/`.

## 1. Tiling

Input: `demo/matmul.mlir`

Output: `01_tiled.mlir`

This step runs the transform schedule from `transforms/matmul_tile.mlir`. It
tiles `linalg.matmul` into 8x8x8 tiles so that individual matmul operations
match the systolic array size.

## 2. Padding

Input: `01_tiled.mlir`

Output: `02_padded.mlir`

This step runs the transform schedule from `transforms/matmul_pad.mlir`. It pads
small or boundary tiles to 8x8 so that the runtime systolic array always receives
fixed-size matrices.

## 3. Bufferization

Input: `02_padded.mlir`

Output: `03_bufferized.mlir`

`--one-shot-bufferize="bufferize-function-boundaries"` converts tensor-level IR
to memref-level IR. After this step, data is represented as memory buffers
instead of SSA tensor values. `--canonicalize` and `--cse` clean up the result
after bufferization.

## 4. Entry Wrapper

Input: `03_bufferized.mlir`

Output: `04_entry_wrapped.mlir`

`--create-c-interface-entry-wrappers` creates a stable external function named
`matmul_entry`. It takes input memrefs and an output memref, calls the original
`matmul`, and copies the result into the provided output buffer. The wrapper gets
`llvm.emit_c_interface` so that C code can call
`_mlir_ciface_matmul_entry`.

## 5. Systolic Conversion

Input: `04_entry_wrapped.mlir`

Output: `05_systolic_memref.mlir`

`--convert-linalg-matmul-to-systolic` replaces matching `linalg.matmul`
operations with `standalone.systolic_matmul`. The pass currently expects an
8x8x8 integer matmul over memref operands:

```text
lhs: memref<8x8xi8>
rhs: memref<8x8xi8>
acc: memref<8x8xi32>
```

## 6. Systolic Call Lowering

Input: `05_systolic_memref.mlir`

Output: `06_systolic_call.mlir`

`--lower-systolic-to-func-call` replaces `standalone.systolic_matmul` with a
regular `func.call @systolic_matmul_8x8`. This function is implemented in the
interface file `interface/interface.c`.

## 7. LLVM Dialect Lowering

Input: `06_systolic_call.mlir`

Output: `07_llvm.mlir`

The remaining MLIR dialects are lowered to the LLVM dialect. In this step,
`linalg`, `scf`, `cf`, `arith`, `index`, `func`, and `memref` are gradually
replaced with low-level LLVM-compatible operations.

## 8. LLVM IR Translation

Input: `07_llvm.mlir`

Output: `08_llvm.ll`

`mlir-translate --mlir-to-llvmir` converts the MLIR LLVM dialect to regular LLVM
IR. This file can then be compiled with `llc` or a compatible LLVM toolchain.
