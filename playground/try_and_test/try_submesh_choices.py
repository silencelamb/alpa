from alpa.pipeline_parallel.stage_construction import get_submesh_choices

num_hosts = 128
num_devices_per_host = 8
# "all" "power_of_two" "small_power_of_two"
# space = "all"
# space = "all"
space = "small_power_of_two"
print(space)
print(get_submesh_choices(num_hosts, num_devices_per_host, space))
space = "all"
print(space)
print(get_submesh_choices(num_hosts, num_devices_per_host, space))
space = "power_of_two"
print(space)
print(get_submesh_choices(num_hosts, num_devices_per_host, space))

