# python benchmark.py --suite  gpt.grid_search_auto --num-hosts 4 --num-devices-per-host 8 --no-separate-process --only-mapping
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 8 --num-devices-per-host 8 --no-separate-process --only-mapping


# config parallel, anylytical model gpu
python benchmark.py --suite  gpt.config_test --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model  --rst_folder tmp_a100_gpu_analytical


# config parallel, anylytical model  wsc, 1x8
python benchmark.py --suite  gpt.config_test --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --rst_folder tmp_wsc_perf_15GB_fp16

# wsc search
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_perf_15GB_fp16


# config parallel, anylytical model, wsc manual, 2x4

python benchmark.py --suite  gpt.wsc_config_test --num-hosts 2 --num-devices-per-host 4 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_perf_15GB_fp16

python benchmark.py --suite  gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_perf_15GB_fp16


# config parallel, anylytical model, wsc manual, MLP
python benchmark.py --suite  mlp.wsc_config_test --num-hosts 1 --num-devices-per-host 4 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc --force_use_fp16 --rst_folder tmp_wsc_mlp_perf_15GB_fp16


# config parallel, anylytical model gpu
python benchmark.py --suite  gpt.config_test --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model  --rst_folder tmp_a100_gpu_analytical

# config parallel, anylytical model WSC
python benchmark.py --suite  gpt.wsc_config_test --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsc  --rst_folder tmp_wsc_perf_15GB_fp16
