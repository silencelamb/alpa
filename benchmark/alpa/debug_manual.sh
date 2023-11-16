python benchmark.py --suite gpt.wsc_perf_debug --num-hosts 1 --num-devices-per-host 7 --use-greedy-collective-cost --hardware wsgpu --rst_folder tmp_wsgpu_debug_7

# python benchmark.py --suite gpt.wsc_perf_debug --num-hosts 1 --num-devices-per-host 8 --use-greedy-collective-cost --hardware wsgpu --rst_folder tmp_wsgpu_debug_8

# python benchmark.py --suite gpt.wsc_perf_debug --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --hardware wsgpu --rst_folder tmp_wsgpu_debug_tp24