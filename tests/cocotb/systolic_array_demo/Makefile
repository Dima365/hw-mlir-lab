TOPLEVEL_LANG ?= verilog
SIM ?= verilator

REPO_ROOT := $(abspath ../../..)
export REPO_ROOT

VERILOG_SOURCES := $(REPO_ROOT)/ip/systolic_array_demo/array.sv
TOPLEVEL := array
MODULE := test_socket_array

EXTRA_ARGS += -GN=8 -GDATA_W=8 -GACC_W=32
EXTRA_ARGS += --trace-fst --trace-depth 99 --trace-max-array 1024 --trace-max-width 1024
VERILATOR_TRACE ?= 1

include $(shell cocotb-config --makefiles)/Makefile.sim
