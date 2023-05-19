cd zhanghaichao/lab1/alpa
pip3 install -e ".[dev]"
cd /zhanghaichao/lab1/alpa/benchmark/alpa
ray stop --force
ray start --head
python  benchmark.py --suite  gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model  --rst_folder tmp_22_gpu_analytical

# python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --hardware gpu --rst_folder  tmp_a100_perf_15GB_200GB_zhc