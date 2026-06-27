TOPLEVEL_LANG ?= verilog
SIM ?= verilator

# Which IP block to simulate. Its toplevel/sources/params come from the
# manifest (ips.yaml) via ips.py, so this Makefile is generic across IPs.
IP ?= matmul

REPO_ROOT := $(abspath ../..)
export REPO_ROOT
export IP

TOPLEVEL := $(shell python3 ips.py toplevel $(IP))
VERILOG_SOURCES := $(addprefix $(REPO_ROOT)/,$(shell python3 ips.py sources $(IP)))
MODULE ?= test_socket

# Per-IP build dir so switching IP never reuses another IP's compiled DUT.
# Stays under sim_build/.
SIM_BUILD ?= sim_build/$(IP)

# Drivers and the manifest loader live here; expose them to the harness.
export PYTHONPATH := $(abspath .):$(PYTHONPATH)

EXTRA_ARGS += $(shell python3 ips.py gargs $(IP))
EXTRA_ARGS += --trace-fst --trace-depth 99 --trace-max-array 1024 --trace-max-width 1024
VERILATOR_TRACE ?= 1

include $(shell cocotb-config --makefiles)/Makefile.sim
