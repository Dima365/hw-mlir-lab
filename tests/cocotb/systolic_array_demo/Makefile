TOPLEVEL_LANG ?= verilog
SIM ?= verilator

REPO_ROOT := $(abspath ../../..)

VERILOG_SOURCES := $(REPO_ROOT)/ip/systolic_array_demo/array.sv
TOPLEVEL := array
MODULE := test_socket_array

EXTRA_ARGS += -GN=8 -GDATA_W=8 -GACC_W=32

include $(shell cocotb-config --makefiles)/Makefile.sim
