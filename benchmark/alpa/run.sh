# python benchmark.py --suite  gpt.grid_search_auto --num-hosts 4 --num-devices-per-host 8 --no-separate-process --only-mapping
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 8 --num-devices-per-host 8 --no-separate-process --only-mapping


# config parallel, anylytical model  
python benchmark.py --suite  gpt.config --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model  --rst_folder tmp_a100_gpu_analytical

