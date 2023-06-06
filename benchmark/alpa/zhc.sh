

# cd /zhanghaichao/lab1/alpa_zhc/build_jaxlib
# python3 build/build.py --enable_cuda --dev_install --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa
# cd dist
# pip3 install -e .
cd /zhanghaichao/lab1/alpa_zhc
pip3 install -e ".[dev]"
cd /zhanghaichao/lab1/alpa_zhc/benchmark/alpa
ray stop --force
ray start --head
# python  benchmark.py --suite  gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model  --rst_folder tmp_22_gpu_analytical

python  benchmark.py --suite  mlp.wsc_config_test --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model  --rst_folder tmp_22_gpu_analytical

# python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --hardware gpu --rst_folder  tmp_a100_perf_15GB_200GB_zhc2023