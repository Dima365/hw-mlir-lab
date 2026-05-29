# Compile Pipeline

Этот файл описывает шаги из `pipelines/compile_pipeline.sh`.

`pipelines/compile_pipeline.sh` собирает исполняемый файл из результата MLIR pipeline и
C driver. Его вызывают `demo/run.sh` и тесты из `tests/matmul/build.py`.

Интерфейс:

```bash
./pipelines/compile_pipeline.sh <llvm-ir> <main.c> <app> <object-dir>
```

## 1. Runtime Object

Вход: `simulator/interface.c`

Выход: `<object-dir>/interface.o`

C runtime компилируется в object-файл. В нем находится реализация
`systolic_matmul_8x8`, которая подключается к Python/Verilator simulator через
Unix socket.

## 2. MLIR Program Object

Вход: `<llvm-ir>`

Выход: `<object-dir>/mlir_program.o`

`llc` компилирует LLVM IR, полученный из MLIR, в object-файл.

## 3. Link App

Входы: `<object-dir>/mlir_program.o`, `<object-dir>/interface.o`, `<main.c>`

Выход: `<app>`

`clang` линкует MLIR object, runtime object, C driver и MLIR runner utils в
готовый executable.
