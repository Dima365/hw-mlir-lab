# Execute Pipeline

Этот файл описывает `pipelines/execute_pipeline.sh`.

Скрипт запускает приложение вместе с Python/Verilator simulator.

Интерфейс:

```bash
./pipelines/execute_pipeline.sh <app>
```

## 1. Cleanup

Удаляются старые socket и ready файлы:

```text
/tmp/systolic_cocotb.sock
/tmp/systolic_cocotb.ready
```

## 2. Start Simulator

Запускается `simulator/verilator.py`. Он создает Unix socket и после `listen`
создает ready-файл.

## 3. Wait Ready

Скрипт ждет появления ready-файла. Если ready-файл не появился за отведенное
число попыток, запуск завершается с ошибкой.

## 4. Run App

После готовности simulator запускается `<app>`. При завершении скрипта simulator
останавливается через cleanup handler.
