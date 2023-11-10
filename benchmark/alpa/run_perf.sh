# Bellow is 6x4 Wafer Scale GPU
# close offload for gpt
python benchmark.py --suite gpt.wsc_perf --num-hosts 6 --num-devices-per-host 4 --hardware wsgpu --rst_folder tmp_wsgpu
# open offload for gpt
python benchmark.py --suite gpt.wsc_perf --offload --num-hosts 6 --num-devices-per-host 4 --hardware wsgpu --rst_folder tmp_wsgpu 
# close offload for bert
python benchmark.py --suite bert.wsc_perf --num-hosts 6 --num-devices-per-host 4 --hardware wsgpu --rst_folder tmp_wsgpu
# open offload for bert
python benchmark.py --suite bert.wsc_perf --offload --num-hosts 6 --num-devices-per-host 4 --hardware wsgpu --rst_folder tmp_wsgpu 
# close offload for wresnet
python benchmark.py --suite wresnet.wsc_perf --num-hosts 6 --num-devices-per-host 4 --hardware wsgpu --rst_folder tmp_wsgpu
# open offload for wresnet
python benchmark.py --suite wresnet.wsc_perf --offload --num-hosts 6 --num-devices-per-host 4 --hardware wsgpu --rst_folder tmp_wsgpu 
