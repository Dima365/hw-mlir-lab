# Cocotb Pipeline

Этот файл описывает `pipelines/cocotb_pipeline.sh`.

Скрипт запускает compiled app вместе с cocotb/Verilator testbench для
`ip/systolic_array_demo/array.sv`.

Интерфейс:

```bash
./pipelines/cocotb_pipeline.sh <app>
```

## 1. Start Cocotb

Скрипт вызывает Makefile:

```bash
make -C tests/cocotb/systolic_array_demo
```

В переменных окружения передаются:

```text
APP       путь к compiled app
REPO_ROOT корень репозитория
```

## 2. Socket Bridge

Cocotb testbench создает Unix socket и ready-файл:

```text
/tmp/systolic_cocotb.sock
/tmp/systolic_cocotb.ready
```

После этого cocotb запускает `<app>`. Runtime function `systolic_matmul_8x8`
из `interface/interface.c` подключается к этому socket.

## 3. Drive RTL

Для каждого matmul request cocotb:

- читает `i8` матрицы A и B;
- читает `i32` аккумулятор C;
- загружает A, B и C в `a_flat`, `b_flat` и `c_in_flat`;
- подает `start`;
- ждет `done`;
- читает `c_out_flat`;
- возвращает `i32` результат обратно в app.
