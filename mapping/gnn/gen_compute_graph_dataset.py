import os
import json
jax_json_path = ""


def gen_computer_graph(json_file, primitve_num_dict, dtyep_num_dict):
    with open(json_file, "r") as f:
        jax_json = json.load(f)

    # jax_json = {
    #     "name": "jaxpr0",
    #     "hash": 0,
    #     "eqns": [
    #         {
    #             "invars": [
    #                 {
    #                     "v": 0,
    #                     "b": 0,
    #                     "t": 0
    #                 },
    #                 {
    #                     "v": 1,
    #                     "b": 0,
    #                     "t": 0
    #                 }
    #             ],
    #             "outvars": [
    #                 {
    #                     "v": 2,
    #                     "b": 0,
    #                     "t": 0
    #                 }
    #             ],
    #             "primitive": "add",
    #             "params": {}
    #         },
    #         {
    #             "invars": [
    #                 {
    #                     "v": 2,
    #                     "b": 0,
    #                     "t": 0
    #                 }
    #             ],
    #             "outvars": [
    #                 {
    #                     "v": 3,
    #                     "b": 0,
    #                     "t": 0
    #                 }
    #             ],
    #             "primitive": "reduce_sum",
    #             "params": {
    #                 "axis": [
    #                     0
    #                 ],
    #                 "keepdims": false
    #             }
    #         }
    #     ],
    #     "consts": [
    #         {
    #             "v": 1,
    #             "b": 0,
    #             "t": 0
    #         }
    #     ]
    # }


json_files = [os.path.join(jax_json_path, f) for f in os.listdir(jax_json_path) if f.endswith('.json')]

with open("primitive_num_dict.json", "r") as f:
    primitve_num_dict = json.load(f)

with open("dtype_num_dict.json", "r") as f:
    dtyep_num_dict = json.load(f)

for json_file in json_files:
    graph_data = gen_computer_graph(json_file, primitve_num_dict, dtyep_num_dict)
    