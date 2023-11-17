
export XLA_PYTHON_CLIENT_PREALLOCATE=false

#!/bin/bash

# 定义硬件类型
declare -a hardware_types=("wsgpu" "dojo")

# 定义模型类型和对应的模型大小
declare -A model_sizes
model_sizes["gpt"]="125M 350M 760M 1.3B 2.6B 6.7B 15B"
# model_sizes["bert"]="Base Large LL LLL LLLL"
# model_sizes["wresnet"]="25.56M 44.55M 60.19M 68.88M 126.88M"
# 定义每个模型大小对应的 micro-batchsize
declare -A batchsizes
# gpt
batchsizes["125M"]=24
batchsizes["350M"]=12
batchsizes["760M"]=12
batchsizes["1.3B"]=6
batchsizes["2.6B"]=6
batchsizes["6.7B"]=6
batchsizes["15B"]=1
batchsizes["Base"]=24
batchsizes["Large"]=12
batchsizes["LL"]=6
batchsizes["LLL"]=1
batchsizes["LLLL"]=1
batchsizes["25.56M"]=20
batchsizes["44.55M"]=18
batchsizes["60.19M"]=16
batchsizes["68.88M"]=14
batchsizes["126.88M"]=12

# 循环遍历模型类型和模型大小
for model in "${!model_sizes[@]}"; do
    for size in ${model_sizes[$model]}; do
        # 获取对应的 micro-batchsize
        micro_batchsize=${batchsizes[$size]}
        echo $micro_batchsize
        # 调用脚本
        python benchmark_GA_1dim.py  \
            --hardware wsgpu \
            --num-hosts 1 \
            --num-devices-per-host 24 \
            --model-type "$model" \
            --model-size "$size" \
            --micro-batchsize ${micro_batchsize} \
            --constrain-mem \
            --use-greedy-collective-cost \
            --rst_folder GA_1117/wsgpu_gpt \
            --no-separate-process --only-mapping  --use-analytical-perf-model &
    done
done

# no constrain-mem, offload
# 循环遍历硬件类型
# for hardware in "${hardware_types[@]}"; do
#     # 循环遍历模型类型和模型大小
#     for model in "${!model_sizes[@]}"; do
#         for size in ${model_sizes[$model]}; do
#             # 调用脚本
#             python benchmark_GA_1dim.py  \
#                 --hardware "$hardware" \
#                 --num-hosts 1 \
#                 --num-devices-per-host 24 \
#                 --model-type "$model" \
#                 --model-size "$size" \
#                 --use-offload \
#                 --use-greedy-collective-cost \
#                 --rst_folder tmp_GA_result \
#                 --no-separate-process --only-mapping  --use-analytical-perf-model
#         done
#     done
# done

# constrain-mem, no offload
# 循环遍历硬件类型
# for hardware in "${hardware_types[@]}"; do
#     # 循环遍历模型类型和模型大小
#     for model in "${!model_sizes[@]}"; do
#         for size in ${model_sizes[$model]}; do
#             # 调用脚本
#             s_name=GA_${hardware}_${model}_${size} 
#             tmux new-session -d -s $s_name
#             tmux send-keys -t $s_name 'python benchmark_GA_1dim.py  \\'
#             tmux send-keys -t $s_name '--hardware "$hardware" \\'
#             tmux send-keys -t $s_name '--num-hosts 1 \\'
#             tmux send-keys -t $s_name '--num-devices-per-host 24 \\'
#             tmux send-keys -t $s_name '--model-type ${model} \\'
#             tmux send-keys -t $s_name '--model-size ${size} \\'
#                 --constrain-mem \
#                 --use-greedy-collective-cost \
#                 --rst_folder GA_1117 \
#                 --no-separate-process --only-mapping  --use-analytical-perf-model
#         done
#     done
# done