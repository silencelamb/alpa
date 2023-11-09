
export XLA_PYTHON_CLIENT_PREALLOCATE=false

#!/bin/bash

# 定义硬件类型
declare -a hardware_types=("wsgpu" "dojo")

# 定义模型类型和对应的模型大小
declare -A model_sizes
model_sizes["gpt"]="125M 350M 760M 1.3B 2.6B 6.7B 15B 39B 76B"
model_sizes["bert"]="Tiny Mini Small Medium Base Large LL LLL LLLL"
model_sizes["wresnet"]="25.56M 44.55M 60.19M 68.88M 126.88M"


# constrain-mem, no offload
# 循环遍历硬件类型
for hardware in "${hardware_types[@]}"; do
    # 循环遍历模型类型和模型大小
    for model in "${!model_sizes[@]}"; do
        for size in ${model_sizes[$model]}; do
            # 调用脚本
            python benchmark_GA_1dim.py  \
                --hardware "$hardware" \
                --num-hosts 1 \
                --num-devices-per-host 24 \
                --model-type "$model" \
                --model-size "$size" \
                --constrain-mem \
                --use-greedy-collective-cost \
                --rst_folder tmp_GA_result \
                --no-separate-process --only-mapping  --use-analytical-perf-model
        done
    done
done

# no constrain-mem, offload
# 循环遍历硬件类型
for hardware in "${hardware_types[@]}"; do
    # 循环遍历模型类型和模型大小
    for model in "${!model_sizes[@]}"; do
        for size in ${model_sizes[$model]}; do
            # 调用脚本
            python benchmark_GA_1dim.py  \
                --hardware "$hardware" \
                --num-hosts 1 \
                --num-devices-per-host 24 \
                --model-type "$model" \
                --model-size "$size" \
                --use-offload \
                --use-greedy-collective-cost \
                --rst_folder tmp_GA_result \
                --no-separate-process --only-mapping  --use-analytical-perf-model
        done
    done
done