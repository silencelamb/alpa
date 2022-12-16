# /bin/bash
# sed -i 's/self.use_analytical_perf_model = False/self.use_analytical_perf_model = True/' /code/alpa/alpa/global_env.py
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 1 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 4 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 2 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 4 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 8 --num-devices-per-host 8 --no-separate-process --only-mapping --use-analytical-perf-model --rst_folder tmp_a100_perf_15GB_25Gbps
ray stop --force
ray start --head
# sed -i 's/self.use_analytical_perf_model = True/self.use_analytical_perf_model = False/' /code/alpa/alpa/global_env.py
