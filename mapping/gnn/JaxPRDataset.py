import json
import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import remove_self_loops


class JaxPRDataset(InMemoryDataset):
    r""".

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    shape_len = 4
    pad_num = 0
    max_in_vars = 3
    max_out_vars = 2

    # primitive 1
    # in vars 3 x (dtype 1 + shape 4) = 15
    # out vars 2 x (dtype 1 + shape 4) = 10
    # TODO params ?

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):

        assert split in ['train', 'val', 'test']
        json_file = osp.join(root, "primitive_num_dict.json")
        if osp.exists(json_file):
            with open(json_file, "r") as f:
                self.primitive_num_dict = json.load(f)
        else:
            self.primitive_num_dict = {}

        json_file = osp.join(root, "dtype_num_dict.json")
        if osp.exists(json_file):
            with open(json_file, "r") as f:
                self.dtype_num_dict = json.load(f)
        else:
            self.dtype_num_dict = {}

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])



    @property
    def raw_file_names(self) -> List[str]:
        json_files = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith('.json')]
        return json_files
    
    @property
    def processed_file_names(self) -> str:
        return ['train.pt']
    
    def convert_json_to_pyg_data(self, json_file: str) -> Data:
        """
        """
        with open(json_file, "r") as f:
            jax_json = json.load(f)
        node_features = []
        edge_index = []
        vars_dict = jax_json["vars"]
        eqns = jax_json["eqns"] 
        for cur_node_id, eqn in enumerate(eqns):
            cur_feat = []
            # encode primitive to number
            primitive = eqn["primitive"]
            primitive_num = self.primitive_num_dict.get(primitive, len(self.primitive_num_dict))
            self.primitive_num_dict[primitive] = primitive_num
            cur_feat.append(primitive_num)

            # encode invars info
            invars = eqn["invars"]
            if len(invars) > self.max_in_vars:
                print(f'========Warning==========: invars num: {len(invars)}, larger than {self.max_in_vars} !!!!!')
                print(eqn)

            for var in invars:
                vars_info = vars_dict[var]
                # encode dtype to number
                dtype = vars_info["dtype"]
                dtype_num = self.dtype_num_dict.get(dtype, len(self.dtype_num_dict))
                self.dtype_num_dict[dtype] = dtype_num
                cur_feat.append(dtype_num)

                # encode shape info
                shape = vars_info["shape"]
                # padding to fix shape_len
                shape =  shape + [self.pad_num] * (self.shape_len - len(shape))
                cur_feat.extend(shape)

                # add edge info
                if "node_id" in vars_info:
                    edge_index.append([vars_info["node_id"], cur_node_id])
            # padding to fix length
            padding = [self.pad_num] * (1+self.shape_len) * (self.max_in_vars - len(invars))
            cur_feat.extend(padding)
            
            # encode outvars info
            outvars = eqn["outvars"]
            if len(outvars) > self.max_out_vars:
                print(f'========Warning==========: invars num: {len(outvars)}, larger than {self.max_out_vars} !!!!!')
                print(eqn)

            for var in outvars:
                vars_info = vars_dict[var]
                # endcode dtype to number
                dtype = vars_info["dtype"]
                dtype_num = self.dtype_num_dict.get(dtype, len(self.dtype_num_dict))
                self.dtype_num_dict[dtype] = dtype_num
                cur_feat.append(dtype_num)

                # encode shape info
                shape = vars_info["shape"]
                shape =  shape + [self.pad_num] * (self.shape_len - len(shape))
                cur_feat.extend(shape)
            # padding to fix length
            padding = [self.pad_num] * (1+self.shape_len) * (self.max_out_vars - len(outvars))
            cur_feat.extend(padding)

            ### TODO encode params info
            # 这里怎么处理比较好？？ dot_general "dimension_numbers": [[[3],[3]], [[0, 2], [0, 2]] ]
            # conv的参数呢？？
            # reshape "new_sizes": [8, 1024, 32, 80]
            # params = eqn["params"]

            node_features.append(cur_feat)

        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        data = Data(edge_index=edge_index, x=node_features, y=node_features)
        return data
        
    
    def process(self):
        data_list = []
        for json_file in self.raw_file_names:
            import pdb; pdb.set_trace()
            data = self.convert_json_to_pyg_data(json_file)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
        json_file = osp.join(self.root, "dtype_num_dict.json")
        with open(json_file, "w") as f:
            json.dump(self.dtype_num_dict, f, indent=4)

        json_file = osp.join(self.root, "primitive_num_dict.json")
        with open(json_file, "w") as f:
            json.dump(self.primitive_num_dict, f, indent=4)