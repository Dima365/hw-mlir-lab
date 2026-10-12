.PHONY: docker-build standalone demo demo-qadd-relu demo-conv-requant demo-resnet-blocks lower-resnet18-blocks test shell docker-config clean-generated clean-standalone

docker-build:
	docker compose build

standalone:
	docker compose run --rm dev ./pipelines/build_standalone.sh

demo: standalone
	docker compose run --rm dev ./demo/run.sh

demo-qadd-relu: standalone
	docker compose run --rm dev ./demo/run_qadd_relu.sh

demo-conv-requant: standalone
	docker compose run --rm dev ./demo/run_conv_requant.sh

demo-resnet-blocks: standalone
	docker compose run --rm dev ./demo/run_resnet_blocks.sh

lower-resnet18-blocks: standalone
	docker compose run --rm dev ./projects/resnet18/lower_resnet18_blocks.sh

test: standalone
	docker compose run --rm dev python tests/run_matmul_tests.py

shell:
	docker compose run --rm dev bash

docker-config:
	docker compose config

clean-generated:
	rm -rf build
	rm -f tests/cocotb/dump.fst
	rm -f tests/cocotb/dump.fst.hier
	rm -f tests/cocotb/results.xml
	rm -rf tests/cocotb/sim_build

clean-standalone:
	rm -rf standalone/build
