
# wsc search
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_perf_15GB_fp16


# config parallel, anylytical model, wsc manual, 2x4

python benchmark.py --suite  gpt.wsc_config_test --num-hosts 2 --num-devices-per-host 4 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_perf_15GB_fp16

python benchmark.py --suite  gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_perf_15GB_fp16

# config parallel, anylytical model WSC
python benchmark.py --suite  gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc  --rst_folder tmp_wsc_perf_15GB_fp16

python benchmark.py --suite  gpt.wsc_config_test --num-hosts 2 --num-devices-per-host 4 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc  --rst_folder tmp_wsc_perf_15GB_fp16

# python benchmark.py --suite gpt.wsc_config_test --num-hosts 5 --num-devices-per-host 4 --no-separate-process --only-mapping --hardware dojo --use-analytical-perf-model --use-greedy-collective-cost --rst_folder tmp_wsc_perf_dojo
# python benchmark.py --suite gpt.wsc_config_test --num-hosts 5 --num-devices-per-host 4 --no-separate-process --only-mapping --hardware wsgpu --use-analytical-perf-model --use-greedy-collective-cost --rst_folder tmp_wsc_perf_wsgpu

python benchmark.py --suite gpt.wsc_config_test --num-hosts 5 --num-devices-per-host 4 --no-separate-process --only-mapping --hardware wsc --use-analytical-perf-model --use-greedy-collective-cost --rst_folder tmp_wsc_perf_tx8_tacos

# 遗传算法 (Genetic Algorithm, GA) 
python benchmark_GA_1x4.py --suite gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 4 --no-separate-process --only-mapping --hardware wsc --use-analytical-perf-model --rst_folder tmp_gpu_analytical_GA_GPT

python benchmark_GA_1dim.py --suite gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 20 --no-separate-process --only-mapping --hardware wsc --use-analytical-perf-model --rst_folder tmp_wsc_perf_tx8_20


# Below all is GPU
# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 1 --num-devices-per-host 1  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 1 --num-devices-per-host 2  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 1 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 2 --num-devices-per-host 4  --rst_folder tmp_exp0


# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 4 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 4 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 8 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_manual --num-hosts 15 --num-devices-per-host 8  --rst_folder tmp_exp0


# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 1 --num-devices-per-host 1  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 1 --num-devices-per-host 2  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 1 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 2 --num-devices-per-host 4  --rst_folder tmp_exp0


# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 2 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 4 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  gpt.perf_test_auto --num-hosts 8 --num-devices-per-host 8  --rst_folder tmp_exp0



# python benchmark.py --suite  bert.perf_test_manual --num-hosts 1 --num-devices-per-host 1  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_manual --num-hosts 1 --num-devices-per-host 2  --rst_folder tmp_exp0


# python benchmark.py --suite  bert.perf_test_manual --num-hosts 1 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_manual --num-hosts 2 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_manual --num-hosts 4 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_manual --num-hosts 4 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_manual --num-hosts 8 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_manual --num-hosts 15 --num-devices-per-host 8  --rst_folder tmp_exp0




# python benchmark.py --suite  bert.perf_test_auto --num-hosts 1 --num-devices-per-host 1  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_auto --num-hosts 1 --num-devices-per-host 2  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_auto --num-hosts 1 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_auto --num-hosts 2 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_auto --num-hosts 2 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_auto --num-hosts 4 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  bert.perf_test_auto --num-hosts 8 --num-devices-per-host 8  --rst_folder tmp_exp0




# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 1 --num-devices-per-host 1  --rst_folder tmp_exp0

# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 1 --num-devices-per-host 2  --rst_folder tmp_exp0

# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 1 --num-devices-per-host 4  --rst_folder tmp_exp0

# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 2 --num-devices-per-host 4  --rst_folder tmp_exp0


# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 2 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 4 --num-devices-per-host 8  --rst_folder tmp_exp0

# python benchmark.py --suite  wresnet.perf_test_auto --num-hosts 8 --num-devices-per-host 8  --rst_folder tmp_exp0
