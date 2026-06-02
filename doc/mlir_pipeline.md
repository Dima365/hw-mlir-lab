# MLIR Pipeline

Этот файл описывает шаги из `pipelines/mlir_pipeline.sh`. По умолчанию входной файл -
`demo/matmul.mlir`, а промежуточные результаты записываются в
`build/mlir-pipeline/`.

## 1. Tiling

Вход: `demo/matmul.mlir`

Выход: `01_tiled.mlir`

Запускается transform schedule из `transforms/matmul_tile.mlir`. Он разбивает
`linalg.matmul` на тайлы размера 8x8x8, чтобы отдельные matmul-операции
соответствовали размеру systolic array.

## 2. Padding

Вход: `01_tiled.mlir`

Выход: `02_padded.mlir`

Запускается transform schedule из `transforms/matmul_pad.mlir`. Он дополняет
маленькие или краевые тайлы до 8x8, чтобы runtime systolic array всегда получал
матрицы фиксированного размера.

## 3. Bufferization

Вход: `02_padded.mlir`

Выход: `03_bufferized.mlir`

`--one-shot-bufferize="bufferize-function-boundaries"` переводит tensor-level IR
в memref-level IR. После этого данные представлены как буферы памяти, а не как
SSA tensor values. `--canonicalize` и `--cse` чистят результат после
bufferization.

## 4. Entry Wrapper

Вход: `03_bufferized.mlir`

Выход: `04_entry_wrapped.mlir`

`--create-c-interface-entry-wrappers` создает стабильную внешнюю функцию
`matmul_entry`. Она принимает входные memref и выходной memref, вызывает
исходную `matmul`, а затем копирует результат в переданный выходной буфер.
На wrapper добавляется `llvm.emit_c_interface`, чтобы C-код мог вызывать
`_mlir_ciface_matmul_entry`.

## 5. Systolic Conversion

Вход: `04_entry_wrapped.mlir`

Выход: `05_systolic_memref.mlir`

`--convert-linalg-matmul-to-systolic` заменяет подходящие `linalg.matmul`
операции на `standalone.systolic_matmul`. Сейчас pass ожидает 8x8x8 integer
matmul на memref-операндах:

```text
lhs: memref<8x8xi8>
rhs: memref<8x8xi8>
acc: memref<8x8xi32>
```

## 6. Systolic Call Lowering

Вход: `05_systolic_memref.mlir`

Выход: `06_systolic_call.mlir`

`--lower-systolic-to-func-call` заменяет `standalone.systolic_matmul` на
обычный `func.call @systolic_matmul_8x8`. Эта функция реализована в runtime
файле `interface/interface.c`.

## 7. LLVM Dialect Lowering

Вход: `06_systolic_call.mlir`

Выход: `07_llvm.mlir`

Оставшиеся MLIR dialects опускаются к LLVM dialect. На этом шаге `linalg`,
`scf`, `cf`, `arith`, `index`, `func` и `memref` постепенно заменяются на
низкоуровневые LLVM-compatible операции.

## 8. LLVM IR Translation

Вход: `07_llvm.mlir`

Выход: `08_llvm.ll`

`mlir-translate --mlir-to-llvmir` переводит MLIR LLVM dialect в обычный LLVM IR.
Этот файл уже можно компилировать через `llc` или совместимый LLVM toolchain.
