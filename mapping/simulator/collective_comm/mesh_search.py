import argparse
import random
from dataclasses import dataclass
from collections import deque
from enum import Enum
import pickle

from alpa.global_env import get_global_config

GLOBAL_CONFIG = get_global_config()

class Collective(Enum):
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_REDUCE = 3
    ALL_TO_ALL = 4
    
@dataclass
class Chunk:
    chunk_id: int
    rank_id: int

    def __str__(self):
        return f"({self.chunk_id}, {self.rank_id})"
    
class MeshNode(object):
    def __init__(self, idx: tuple, rank: int, links, coll: Collective, data=None):
        self.idx = idx              # (i, j) index in the node map
        self.rank = rank            # rank id, index start from 1
        self.links = links          # neighbour nodes idx in the mesh
        self.collective = coll      # collective type

        self.postcondition = set()  # the postcondition chunk set
        self.chunks = set()         # set of Chunks
        self.state = False          # True if the node reaches postcondition
        self.sent_num = 0           # concurrent sent chunks in a single timestep
        self.recv_num = 0           # concurrent received chunks in a single timestep
        self.link_max = len(links)
        # self.alpha = alpha # link latency
        # self.betas = betas # double side link bandwidth
        # self.data_size = 1024       # 1KB

    def add_chunk(self, chunk):
        self.chunks.add(chunk)
    
    def del_chunk(self, chunk):
        self.chunks.remove(chunk)

    def is_finished(self):
        if self.collective == Collective.ALL_GATHER:
            return self.chunks == self.postcondition
        elif self.collective == Collective.REDUCE_SCATTER:
            return (self.postcondition in self.reduce_chunks)
    
    def reset_links(self):
        self.sent_num = 0
        self.recv_num = 0
    
    def __str__(self):
        return f"Node {self.idx}, chunks {self.chunks}, links {self.links}"

class ReduceNode(MeshNode):
    def __init__ (self, idx, rank, links, coll):
        super().__init__(idx, rank, links, coll)
        self.reduce_chunks = []
    
    def add_chunks(self, idx, chunks: set) -> bool:
        self.reduce_chunks[idx][0] = chunks | self.reduce_chunks[idx][0]

class Mesh(object):
    def __init__ (self, m: int, n: int, coll: Collective):
        self.row_len = m
        self.col_len = n
        self.rank_num = m * n
        self.collective = coll
        self.rank2idx = {}
        self.max_links = 2 * (m * (n - 1) + (m - 1) * n)
        self.finished = 0
        self.chunk_sum = sum([x + 1 for x in range(m * n)])
        self.sent_dict = {}
        # self.alpha = alpha      # us
        # self.beta = beta        # us/MB
    def init_mesh(self):
        self.init_node_map()
        self.init_precondtion()
        self.init_postcondition()

    def init_node_map(self):
        self.node_map = self.create_node_map(self.row_len, self.col_len)

    def manhattan_dist(self, src_node, dst_node):
        return abs(src_node[0] - dst_node[0]) + abs(src_node[1] - dst_node[1])

    def __str__(self):
        mesh_str = "[" + "1-" * (self.col_len - 1) + "1]\n"
        mesh_str *= self.row_len
        return mesh_str

    # ([Chunk c, NPU n) pair
    def init_precondtion(self):
        if self.collective == Collective.ALL_GATHER:
            for i, node in enumerate(self.node_map.values()):
                node.chunks.add(i + 1)

        elif self.collective == Collective.REDUCE_SCATTER:
            for i, node in enumerate(self.node_map.values()):
                for j in range(1, self.rank_num + 1):
                    node.reduce_chunks.append([{i + 1}, j])
                # print(node.reduce_chunks)

    def init_postcondition(self):
        if self.collective == Collective.ALL_GATHER:
            for rank_id, node in enumerate(self.node_map.values()):
                for chunk_id in range(1, self.rank_num + 1):
                    node.postcondition.add(chunk_id)

        elif self.collective == Collective.REDUCE_SCATTER:
            reduced_chunks = {chunk_id + 1 for chunk_id, _ in enumerate(self.node_map.keys())}
            for i, node in enumerate(self.node_map.values()):
                node.postcondition = [reduced_chunks, i + 1]
                # print(node.postcondition)
                
    def create_node_map(self, m, n):
        node_map = {}
        rank = 1
        for i in range(m):
            for j in range(n):
                idx = (i, j)
                links = []
                # Add top, left, down, right nodes if exists.
                if i > 0:
                    links.append((i - 1, j))
                if j > 0:
                    links.append((i, j - 1))
                if i < m - 1:
                    links.append((i + 1, j))
                if j < n - 1:
                    links.append((i, j + 1))
                if self.collective == Collective.ALL_GATHER:
                    node_map[idx] = MeshNode(idx, rank, links, self.collective)
                elif self.collective == Collective.REDUCE_SCATTER:
                    node_map[idx] = ReduceNode(idx, rank, links, self.collective)
                self.rank2idx[rank] = idx
                rank += 1
        return node_map
    
    def update_sent_dict(self, timestep, chunk, src_node, dst_node):     
        if timestep not in self.sent_dict:
            self.sent_dict[timestep] = [(chunk, src_node, dst_node)]
        else:
            self.sent_dict[timestep].append((chunk, src_node, dst_node))
        
    # def find_ring_path(self):
    #     self.ring_path = []
    #     matrix_size = self.row_len
    #     current_position = (0, 0)

    #     while matrix_size > 0:
    #         (row, col) = current_position
    #         cur_ring = []
    #         # Traverse Right
    #         for j in range(col, col + matrix_size):
    #             cur_ring.append((row, j))
    #         # Traverse Down
    #         for i in range(row + 1, row + matrix_size):
    #             cur_ring.append((i, col + matrix_size - 1))
    #         # Traverse Left
    #         if matrix_size > 1:
    #             for j in range(col + matrix_size - 2, col - 1, -1):
    #                 cur_ring.append((row + matrix_size - 1, j))
    #         # Traverse Up
    #         if matrix_size > 1:
    #             for i in range(row + matrix_size - 2, row, -1):
    #                 cur_ring.append((i, col))
    #         current_position = (row + 1, col + 1)
    #         matrix_size -= 2

    #         self.ring_path.append(cur_ring)
    #     return self.ring_path

    def is_finished(self):
        for node in self.node_map.values():
            if node.state == False:
                return False
        return True

    def print_cur_chunks(self):
        if self.collective == Collective.ALL_GATHER:
            for node in self.node_map.values():
                print(f"{node.rank}: {sorted(node.chunks)} {node.state}")
        elif self.collective == Collective.REDUCE_SCATTER:
            for node in self.node_map.values():
                print(f"{node.rank}: {node.reduce_chunks} {node.state}")
         
    def clear_cur_chunks(self):
        if self.collective == Collective.REDUCE_SCATTER:
            for node in self.node_map.values():
                node.chunks = node.chunks & node.postcondition
    
    def print_preconditions(self):
        print("Each node pre condition")
        for node in self.node_map.values():
            print(f"{node.rank}: {sorted(node.chunks)}")

    def print_postconditions(self):
        print("Each node post condition")
        for node in self.node_map.values():
            print(f"{node.rank}: {sorted(node.postcondition)}")


# compute all gather cost
def compute_all_gather_cost(mesh: Mesh):
    # if mesh.row_len < 3 and (mesh.row_len % 2 == 0 or mesh.col_len % 2 == 0):
    #     return mesh.rank_num - 1
    queue = deque(mesh.node_map.values())
    time_steps = 0
    total_cost = 0
    # in each time step each directed link can be only used once
    while queue: 
        count = 0
        # print(f"Time Step {time_steps}")
        node_visited = []
        for _ in range(len(queue)):  # Go only through nodes present in the queue at current timestep
            node = queue.popleft()
            requsted_chunks = list(node.postcondition.difference(node.chunks))
            while len(requsted_chunks) > 0:
                if node.recv_num == len(node.links) or node.sent_num == len(node.links):
                    break
                chunk = requsted_chunks.pop()
                candidates = []
                for source_idx in node.links:
                    source_node = mesh.node_map[source_idx]
                    if chunk not in source_node.chunks:
                        continue
                    candidates.append(source_idx)
                if len(candidates) > 0:
                    match_cand_idx = random.choice(candidates) # randomly select a candidate
                    match_cand = mesh.node_map[match_cand_idx]
                    match_cand.sent_num += 1
                    node.recv_num += 1
                    node.add_chunk(chunk)
                    count += 1
                    mesh.update_sent_dict(time_steps, chunk, match_cand.rank, node.rank)
                    if node.idx not in node_visited:
                        node_visited.append(node.idx)
                    if match_cand_idx not in node_visited:
                        node_visited.append(match_cand_idx)                
            if node.is_finished():
                node.state = True
                node.chunks = node.postcondition & node.chunks
            
            if node.state == False:
                queue.append(node)
            # push unfinished nodes to queue
            for idx in node.links:
                neighbor = mesh.node_map[idx]
                if neighbor.state == False and neighbor not in queue:
                    queue.append(neighbor)
        
        for idx in node_visited:
            mesh.node_map[idx].reset_links()
        if count > 0:
            time_steps += 1
            # print(f"links transfering at the same time: {count}")
            # total_cost += mesh.alpha + mesh.beta
    return time_steps

def simulate_reduce_scatter(mesh: Mesh):
    # generate all-gather sent dict
    compute_all_gather_cost(mesh)
    mesh_timesteps = list(mesh.sent_dict.keys())
    reduce_scatter_dict = {}
    time_step = 0
    for t in mesh_timesteps[::-1]:
        for sent_items in mesh.sent_dict[t][::-1]:
            if time_step not in reduce_scatter_dict:
                reduce_scatter_dict[time_step] = [(sent_items[0], sent_items[2], sent_items[1])]
            else:
                reduce_scatter_dict[time_step].append((sent_items[0], sent_items[2], sent_items[1]))
        time_step += 1
    print(reduce_scatter_dict)
    new_mesh = Mesh(M, N, Collective.REDUCE_SCATTER)
    new_mesh.init_mesh()
    new_mesh.print_cur_chunks()
    for t in reduce_scatter_dict.keys():
        for sent_items in reduce_scatter_dict[t]:
            chunk_id, src_rank, dst_rank = sent_items
            src_idx = new_mesh.rank2idx[src_rank]   # src node index
            dst_idx = new_mesh.rank2idx[dst_rank]   # dst node index
            sent_chunks = new_mesh.node_map[src_idx].reduce_chunks[chunk_id - 1][0] # set of chunks
            new_mesh.node_map[dst_idx].add_chunks(chunk_id - 1, sent_chunks)
    for node in new_mesh.node_map.values():
        if node.is_finished():
            node.state = True
    new_mesh.print_cur_chunks()

# reduce-scatter is an inverse all-gather
def compute_reduce_scatter_cost(mesh: Mesh):
    return compute_all_gather_cost(mesh)

def compute_all_reduce_cost(mesh: Mesh):
    timesteps = compute_all_gather_cost(mesh)
    return 2 * timesteps
    

class MeshCollectiveCostDatabase:
    """A database that stores collective search results for multiple sub-meshes .
    Key:   (collective: int, mesh_shape: (int, int))
    Value: timesteps: int
    """
    def __init__(self, data=None):
        self.data = data or {}

    def query(self, cluster_key, mesh_shape):
        key = (cluster_key, mesh_shape)
        return self.data[key]

    def update_one_mesh(self, collective, mesh_shape, mesh_result):
        key = (collective, mesh_shape)
        if key not in self.data:
            self.data[key] = mesh_result
        else:
            self.data[key].update(mesh_result)
    

    def update(self, data):
        self.data = data
        # for ((coll, mesh_shape), timesteps) in new_database.data.items():
            # self.update_one_mesh(, mesh_shape, mesh_result)

    def insert_dummy_mesh_result(self, cluster_key, mesh_shape):
        """Insert dummy results for a mesh."""
        key = (cluster_key, mesh_shape)
        assert key not in self.data

        # Copy data from mesh shape (1, 1)
        src_key = (cluster_key, (1, 1))
        assert src_key in self.data
        self.data[key] = self.data[src_key]

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.data, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            new_data = pickle.load(f)
        self.update(new_data)
        # self.update(MeshCollectiveCostDatabase(new_data))
    
    def inspect(self, filename):
        self.load(filename)
        print(list(self.data.items()))

    def __str__(self):
        ret = ""
        for (cluster_key, mesh_shape), value in self.data.items():
            ret += f"cluster_key: {cluster_key}, mesh_shape: {mesh_shape}\n"
            ret += str(value)
        return ret


def generate_database():
    mesh_db = MeshCollectiveCostDatabase()
    collectives = [
                    Collective.ALL_GATHER,
                    Collective.REDUCE_SCATTER, 
                    Collective.ALL_REDUCE
                  ]
    row_num = GLOBAL_CONFIG.wsc_config["analytical_perf_wsc::die_r_num"]
    col_num = GLOBAL_CONFIG.wsc_config["analytical_perf_wsc::die_c_num"]
    for coll in collectives:
        for i in range(1, row_num + 1):
            for j in range(1, col_num + 1):
                mesh = Mesh(i, j, Collective.ALL_GATHER)
                mesh.init_mesh()
                if coll == Collective.ALL_GATHER:
                    timesteps = compute_all_gather_cost(mesh)
                    # print(mesh.sent_dict)
                elif coll == Collective.REDUCE_SCATTER:
                    timesteps = compute_reduce_scatter_cost(mesh)
                else:
                    timesteps = compute_all_reduce_cost(mesh)
                mesh_db.update_one_mesh(coll, (i, j), timesteps)

    mesh_db.save('mesh_coll_cost_database.pkl')
    mesh_db.inspect('mesh_coll_cost_database.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='db')
    parser.add_argument("--method", type=str, default='greedy')
    parser.add_argument("--coll", type=int, default=1)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--col", type=int, default=3)
    parser.add_argument("-v", action='store_true')

    args = parser.parse_args()
    verbose = args.v
    if args.mode == 'online':
        M, N = args.row, args.col
        collective = Collective(args.coll)
        # get global configs

        alpha = 1           # DGX-2 NVlink is 0.7us
        beta = 10           # DGX-2 NVlink is 8us/MB
        data_size = 1       # send 1MB size chunk
        mesh = Mesh(M, N, Collective.ALL_GATHER)
        mesh.init_mesh()
        print(f"Mesh can send/recv at most {mesh.max_links} links in parallel")
        if verbose:
            print("-----------------cur chunks------------------------")
            mesh.print_cur_chunks()
            print("------------- Greedy Search Starts ----------------")
        # cProfile.run("compute_allgather_cost(mesh, collective)")
        timesteps, total_cost = 0, 0

        if collective == Collective.ALL_GATHER:
            timesteps = compute_all_gather_cost(mesh)
        elif collective == Collective.REDUCE_SCATTER:
            timesteps = compute_reduce_scatter_cost(mesh)
        elif collective == Collective.ALL_REDUCE:
            timesteps = compute_reduce_scatter_cost(mesh)
            timesteps += compute_all_gather_cost(mesh)
            print(f"time step {timesteps}")
        if verbose:
            print("------------- Greedy Search Ends ------------------")
            print("cur chunks after search")
            mesh.print_cur_chunks()
            # mesh.print_postconditions()
        # alpha is 1 (us), beta is 10 (us/MB), about 
        # 100GB = 102400MB
        # ring_cost = (M * N - 1) * (alpha + data_size * beta)
        print(f"{collective} Ring need {M * N - 1} steps")
        print(f"Finish {collective} in {timesteps} time steps")
    else:
        # search for all mesh results and store in databaseg
        generate_database()