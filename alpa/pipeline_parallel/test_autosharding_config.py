from stage_construction import get_one_submesh_autosharding_config_choices
from alpa.device_mesh import VirtualPhysicalMesh

# case 1
print("=========case1: 8 device, all=============================")
# Create a virtual submesh with 8 devices
virtual_submesh = VirtualPhysicalMesh( host_ids=[0],
                                      host_info=[{"NodeManagerAddress": "192.128.1.0"}],
                                      head_ip="192.128.1.0",
                                      num_devices_per_host=8
                                      )

# Get the list of logical meshes and autosharding configs
choices = get_one_submesh_autosharding_config_choices(
    virtual_submesh, space="all", batch_size=32)

# Print the resulting choices
for choice in choices:
    print(choice[0].shape, choice[1])

choices = None
    

print("==========case2: 2x8 device, all============================")
# case 2
# Create a virtual submesh with 8 devices
virtual_submesh = VirtualPhysicalMesh( host_ids=[0,1],
                                      host_info=[{"NodeManagerAddress": "192.128.1.0"},
                                                 {"NodeManagerAddress": "192.128.1.0"}],
                                      head_ip="192.128.1.0",
                                      num_devices_per_host=8
                                      )

# Get the list of logical meshes and autosharding configs
choices = get_one_submesh_autosharding_config_choices(
    virtual_submesh, space="all", batch_size=64)

# Print the resulting choices
for choice in choices:
    print(choice[0].shape, choice[1])
choices = None
print("==========case3: 2x8 device, same_as_physical============================")

# case 3
# Create a virtual submesh with 8 devices
virtual_submesh = VirtualPhysicalMesh( host_ids=[0,1],
                                      host_info=[{"NodeManagerAddress": "192.128.1.0"},
                                                 {"NodeManagerAddress": "192.128.1.0"}],
                                      head_ip="192.128.1.0",
                                      num_devices_per_host=8
                                      )

# Get the list of logical meshes and autosharding configs
choices = get_one_submesh_autosharding_config_choices(
    virtual_submesh, space="same_as_physical", batch_size=64)

# Print the resulting choices
for choice in choices:
    print(choice[0].shape, choice[1])
choices = None
print("============case4: 2x8 device, data_parallel_only==========================")

# case 4
# Create a virtual submesh with 8 devices
virtual_submesh = VirtualPhysicalMesh( host_ids=[0,1],
                                      host_info=[{"NodeManagerAddress": "192.128.1.0"},
                                                 {"NodeManagerAddress": "192.128.1.0"}],
                                      head_ip="192.128.1.0",
                                      num_devices_per_host=8
                                      )

# Get the list of logical meshes and autosharding configs
choices = get_one_submesh_autosharding_config_choices(
    virtual_submesh, space="data_parallel_only", batch_size=64)

# Print the resulting choices
for choice in choices:
    print(choice[0].shape, choice[1])
choices = None

print("============case5: 1x8 device, single_node_model_parallel==========================")
# case 5
# Create a virtual submesh with 8 devices
virtual_submesh = VirtualPhysicalMesh( host_ids=[0],
                                      host_info=[{"NodeManagerAddress": "192.128.1.0"}],
                                      head_ip="192.128.1.0",
                                      num_devices_per_host=8
                                      )

# Get the list of logical meshes and autosharding configs
choices = get_one_submesh_autosharding_config_choices(
    virtual_submesh, space="single_node_model_parallel", batch_size=64)

# Print the resulting choices
for choice in choices:
    print(choice[0].shape, choice[1])