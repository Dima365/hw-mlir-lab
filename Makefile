.PHONY: docker-build standalone demo test shell docker-config clean-generated clean-standalone

docker-build:
	docker compose build

standalone:
	docker compose run --rm dev ./pipelines/build_standalone.sh

demo: standalone
	docker compose run --rm dev ./demo/run.sh

test: standalone
	docker compose run --rm dev python tests/run_matmul_tests.py

shell:
	docker compose run --rm dev bash

docker-config:
	docker compose config

clean-generated:
	rm -rf build
	rm -f tests/cocotb/systolic_array_demo/dump.fst
	rm -f tests/cocotb/systolic_array_demo/dump.fst.hier
	rm -f tests/cocotb/systolic_array_demo/results.xml
	rm -rf tests/cocotb/systolic_array_demo/sim_build

clean-standalone:
	rm -rf standalone/build
