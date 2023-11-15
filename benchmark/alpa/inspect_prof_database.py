"""Inspect and edit a profiling database."""
import argparse

from alpa import DeviceCluster, ProfilingResultDatabase
from alpa.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="prof_database_a100.pkl")
    args = parser.parse_args()

    prof_database = ProfilingResultDatabase()
    prof_database.load(args.filename)

    # Do some editing
    #prof_database.insert_dummy_mesh_result("default", (8, 8))
    #prof_database.save(args.filename)

    # Print results
    print("Meshes:")
    print(list(prof_database.data.keys()))

    mesh_result = prof_database.query("default", (1, 8))

    dot_cost_dict = mesh_result.dot_cost_dict
    print(dot_cost_dict.keys())

    fp16_dot_cost = dot_cost_dict[((), 'f16')]
    fp32_dot_cost = dot_cost_dict[((), 'f32')]

    print("fp16 dot cost: ", len(fp16_dot_cost), fp16_dot_cost)
    print("fp32 dot cost: ", len(fp32_dot_cost), fp32_dot_cost)