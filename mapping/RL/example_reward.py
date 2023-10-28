import numpy as np

compute_of_a_microbatch= 100
num_microbatch = 1024
rows, cols = (5, 5)
grid = np.zeros((rows, cols))

constant = compute_of_a_microbatch * num_microbatch

current_step = 0
max_steps = rows * cols
compute_list = []
rect_list = []
rect_id = 0

# compute_list = [100]
# rect_list = [(0, 0, 4, 4)]
# total_latency: 8192.0, 94208.0

# compute_list = [4]*25
# rect_list = [(0, 0, 0, 0)]*5
# total_latency: 4276.4800000000005, 98123.52    # 这个是最好的，也就是说，每个die都有一个计算任务，这样的话，就不需要通信了

compute_list = [20]*5
rect_list = [(0, 0, 0, 4)]*5
# total_latency: 4934.4, 97465.6

rect_num = rect_id
bubble_ratio = rect_num / num_microbatch
stage_latency_list = []
for compute, submesh in zip(compute_list, rect_list):
    left, top, right, bottom = submesh
    die_num = (right - left + 1) * (bottom - top + 1)
    
    communication_ratio = die_num/25
    compute_time = compute / die_num
    stage_time = compute_time * (1+communication_ratio)  # condider communication ratio
    
    stage_latency_list.append(stage_time)
max_stage_lantecy = max(stage_latency_list)
# reshard_comm = 
total_latency = sum(stage_latency_list) + max_stage_lantecy*(num_microbatch-1)
print(f"total_latency: {total_latency}, {constant-total_latency}")