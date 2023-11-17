import argparse
import random
from dataclasses import dataclass
from collections import deque
from enum import Enum
import pickle

class Collective(Enum):
    ALL_GATHER = 0
    REDUCE_SCATTER = 1
    ALL_REDUCE = 2
    ALL_TO_ALL = 3
    

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

    def add_chunk(self, chunk: int):
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
    if mesh.rank_num == 2:
        return mesh.rank_num - 1
    queue = deque(mesh.node_map.values())
    time_steps = 0
    total_cost = 0
    # in each time step each directed link can be only used once
    while queue: 
        count = 0
        # print(f"Time Step {time_steps}")
        node_visited = {}
        send_pair = set()
        for _ in range(len(queue)):  # Go only through nodes present in the queue at current timestep
            dst_node = queue.popleft()
            requsted_chunks = list(dst_node.postcondition.difference(dst_node.chunks))
            while len(requsted_chunks) > 0:
                if dst_node.recv_num == len(dst_node.links):
                    break
                chunk = requsted_chunks.pop()
                candidates = []
                for source_idx in dst_node.links:
                    source_node = mesh.node_map[source_idx]
                    if chunk not in source_node.chunks:
                        continue
                    candidates.append(source_idx)
                if len(candidates) > 0:
                    match_cand_idx = random.choice(candidates) # randomly select a candidate
                    # a link can only be used once in each timestep
                    if (match_cand_idx, dst_node.idx) in send_pair:
                        continue
                    src_node = mesh.node_map[match_cand_idx]
                    if src_node.sent_num == len(src_node.links):
                        continue
                    src_node.sent_num += 1
                    dst_node.recv_num += 1
                    # dst_node.add_chunk(chunk)
                    count += 1
                    mesh.update_sent_dict(time_steps, chunk, src_node.rank, dst_node.rank)
                    send_pair.add((match_cand_idx, dst_node.idx))
                    if dst_node.idx not in node_visited:
                        node_visited[dst_node.idx] = [chunk]
                    else:
                        node_visited[dst_node.idx].append(chunk)
                    if match_cand_idx not in node_visited:
                        node_visited[match_cand_idx] = []
                    # print(f"time: {time_steps} send {chunk} from {src_node.rank} to {dst_node.rank}")
            if dst_node.is_finished():
                dst_node.state = True
                dst_node.chunks = dst_node.postcondition & dst_node.chunks
            
            if dst_node.state == False:
                queue.append(dst_node)
            # push unfinished nodes to queue
            for idx in dst_node.links:
                neighbor = mesh.node_map[idx]
                if neighbor.state == False and neighbor not in queue:
                    queue.append(neighbor)
        # add chunk for all nodes
        for idx in node_visited.keys():
            for chunk in node_visited[idx]:
                mesh.node_map[idx].add_chunk(chunk)
        for idx in node_visited.keys():
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

    def update_one_mesh(self, collective: Collective, mesh_shape: tuple, mesh_cost: int):
        key = (collective, mesh_shape)
        self.data[key] = mesh_cost
    
    def update(self, data):
        self.data = data
        # for ((coll, mesh_shape), timesteps) in new_database.data.items():
            # self.update_one_mesh(, mesh_shape, mesh_result)

    def insert_dummy_mesh_result(self, coll, mesh_shape):
        """Insert dummy results for a mesh."""
        key = (coll, mesh_shape)
        assert key not in self.data

        # Copy data from mesh shape (1, 1)
        src_key = (coll, (1, 1))
        assert src_key in self.data
        self.data[key] = self.data[src_key]

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.data, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            new_data = pickle.load(f)
        self.update(new_data)
    
    def inspect(self, filename):
        self.load(filename)
        print(list(self.data.items()))

    def __str__(self):
        ret = ""
        for (cluster_key, mesh_shape), value in self.data.items():
            ret += f"cluster_key: {cluster_key}, mesh_shape: {mesh_shape}\n"
            ret += str(value)
        return ret


def gen_collective_cost_dict(die_r_num, die_c_num):
    mesh_db = MeshCollectiveCostDatabase()
    collectives = {
        0: Collective.ALL_GATHER,
        1: Collective.REDUCE_SCATTER,
        2: Collective.ALL_REDUCE
    }
    for coll in collectives.keys():
        for i in range(1, die_r_num + 1):
            for j in range(1, die_c_num + 1):
                timesteps = 9999999
                for _ in range(10):
                    mesh = Mesh(i, j, Collective.ALL_GATHER)
                    mesh.init_mesh()
                    if collectives[coll] == Collective.ALL_GATHER:
                        timesteps = min(compute_all_gather_cost(mesh), timesteps)
                    elif collectives[coll] == Collective.REDUCE_SCATTER:
                        timesteps = min(compute_reduce_scatter_cost(mesh), timesteps)
                    else:
                        timesteps = min(compute_all_reduce_cost(mesh), timesteps)
                mesh_db.update_one_mesh(coll, (i, j), timesteps)
    print(mesh_db.data)
    mesh_db.save('mesh_coll_cost_database.pkl')
    return mesh_db.data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='db')
    parser.add_argument("--coll", type=int, default=0)
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
            timesteps = 9999999
            for i in range(10):
                mesh = Mesh(M, N, Collective.ALL_GATHER)
                mesh.init_mesh()
                timesteps = min(compute_all_gather_cost(mesh), timesteps)
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
        # search for all mesh results and store in database
        gen_collective_cost_dict(6, 6)