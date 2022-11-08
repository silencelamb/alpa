# /bin/bash
# sed -i 's/self.use_analytical_perf_model = True/self.use_analytical_perf_model = False/' /code/alpa/alpa/global_env.py
cp prof_database_alpa_slack.pkl prof_database.pkl
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 1 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 2 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 4 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 1 --num-devices-per-host 8 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 2 --num-devices-per-host 8 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 4 --num-devices-per-host 8 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
python benchmark.py --suite  gpt.grid_search_auto --num-hosts 8 --num-devices-per-host 8 --no-separate-process --only-mapping --rst_folder tmp_v100_costmodel
ray stop --force
ray start --head
# sed -i 's/self.use_analytical_perf_model = False/self.use_analytical_perf_model = True/' /code/alpa/alpa/global_env.py
