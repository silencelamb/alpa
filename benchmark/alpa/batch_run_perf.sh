# Bellow is 6x4 Wafer Scale GPU
# close offload for gpt
python benchmark.py --suite gpt.wsc_perf --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsgpu --rst_folder tmp_wsgpu
# open offload for gpt
# python benchmark.py --suite gpt.wsc_perf --offload --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsgpu --rst_folder tmp_wsgpu 
# close offload for bert
python benchmark.py --suite bert.wsc_perf --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsgpu --rst_folder tmp_wsgpu
# open offload for bert
# python benchmark.py --suite bert.wsc_perf --offload --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsgpu --rst_folder tmp_wsgpu 
# # close offload for wresnet
# python benchmark.py --suite wresnet.wsc_perf --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsgpu --rst_folder tmp_wsgpu
# # TODO: 
# # open offload for wresnet
# python benchmark.py --suite wresnet.wsc_perf --offload --num-hosts 6 --num-devices-per-host 4 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware wsgpu --rst_folder tmp_wsgpu 


# Bellow is 5x5 DOJO
# close offload for gpt
python benchmark.py --suite gpt.wsc_perf --num-hosts 5 --num-devices-per-host 5 --hardware dojo --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_dojo
# open offload for gpt
# python benchmark.py --suite gpt.wsc_perf --offload --num-hosts 5 --num-devices-per-host 5 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware dojo --rst_folder tmp_dojo 
# close offload for bert
python benchmark.py --suite bert.wsc_perf --num-hosts 5 --num-devices-per-host 5 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware dojo --rst_folder tmp_dojo
# open offload for bert
# python benchmark.py --suite bert.wsc_perf --offload --num-hosts 5 --num-devices-per-host 5 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware dojo --rst_folder tmp_dojo 
# # close offload for wresnet
# python benchmark.py --suite wresnet.wsc_perf --num-hosts 5 --num-devices-per-host 5 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware dojo --rst_folder tmp_dojo
# # open offload for wresnet
# python benchmark.py --suite wresnet.wsc_perf --offload --num-hosts 5 --num-devices-per-host 5 --use-greedy-collective-cost --no-separate-process --only-mapping --use-analytical-perf-model --hardware dojo --rst_folder tmp_dojo 
