import argparse
import sys
import random
import numpy
from dataclasses import dataclass
from collections import deque
import cProfile

@dataclass
class Chunk:
    chunk_id: int
    rank_id: int

    def __str__(self):
        return f"({self.chunk_id}, {self.rank_id})"
class MeshNode(object):
    def __init__(self, idx, rank, links, alpha, betas, data=None):
        self.idx = idx
        self.rank = rank
        self.links = links # link node idx in the mesh
        self.alpha = alpha # link latency
        self.betas = betas # double side link bandwidth
        self.postcondition = set()  # set of Chunks
        self.precondition = set()   # set of Chunks
        self.chunks = set()         # set of Chunks
        self.data_size = 1024 # 1KB
        self.state = False
        self.sent_num = 0
        self.recv_num = 0

    def add_chunk(self, chunk: Chunk):
        self.chunks.add(chunk)

    # def send_chunk(self, chunk: Chunk):
    #     self.chunks.remove(chunk)
    
    def is_finished(self):
        return self.chunks == self.postcondition
    
    def reset_links(self):
        self.sent_num = 0
        self.recv_num = 0
    
    # def check_state(self):
    #     return self.postcondition == set()

    def __str__(self):
        return f"Node {self.idx}, chunks {self.chunks}, links {self.links}"


class Mesh(object):
    def __init__ (self, m, n, alpha, beta):
        self.row_len = m
        self.col_len = n
        self.rank_num = m * n
        self.alpha = alpha      # us
        self.beta = beta        # us/MB
        self.node_map = self.create_node_map(m, n)
        self.max_links = 2 * (m * (n - 1) + (m - 1) * n)
        self.precondition = set()
        self.postcondition = set()
        self.finished = 0
        self.chunk_sum = sum([x + 1 for x in range(m * n)])

    def __str__(self):
        mesh_str = "[" + "1-" * (self.col_len - 1) + "1]\n"
        mesh_str *= self.row_len
        return mesh_str

    # ([Chunk c, NPU n) pair
    def init_precondtion(self, collective: str):
        if collective == "all_gather":
            for i, node in enumerate(self.node_map.values()):
                self.precondition.add((i + 1, i + 1))
                node.precondition.add(i + 1)
                node.chunks.add(i + 1)

        elif collective == "reduce_scatter":
            for i, node in enumerate(self.node_map.values()):
                for j in range(self.rank_num):
                    self.precondition.add((i + 1, j + 1))
                    node.precondition.add(i + i)
                    node.chunks.add(i + 1)
                # node.precondition.add(()

    def init_postcondition(self, collective: str):
        if collective == "all_gather":
            for i, node in enumerate(self.node_map.values()):
                for j in range(self.rank_num):
                    self.postcondition.add((i + 1, j + 1))
                    node.postcondition.add(j + 1)
                # node.init_postcondition(post_cond)
        elif collective == "reduce_scatter":
            for i, node in enumerate(self.node_map.values()):
                self.postcondition.add((self.chunk_sum, i + 1))
                node.postcondition.add(self.chunk_sum)

            # for i, node in enumerate(self.node_map.values()):
                
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
                node_map[idx] = MeshNode(idx, rank, links, self.alpha, self.beta)
                rank += 1                
        return node_map

    def print_cur_chunks(self):
        for node in self.node_map.values():
            print(f"{node.rank}: {sorted(node.chunks)} {node.state}")
    
    def print_preconditions(self):
        # print(f"Pre:  {sorted(self.precondition)}")
        print("Each node pre condition")
        for node in self.node_map.values():
            print(f"{node.rank}: {sorted(node.precondition)}")

    def print_postconditions(self):
        # print(f"Post: {sorted(self.postcondition)}")
        print("Each node post condition")
        for node in self.node_map.values():
            print(f"{node.rank}: {sorted(node.postcondition)}")

def greedy_search(mesh: Mesh):
    # return all 

    queue = deque(mesh.node_map.values())
    time_steps = 0
    total_cost = 0
    count = 0
                
    # in each time step each directed link can be only used once
    while queue:  # As long as there are nodes in the queue
        count = 0
        print(f"Time Step {time_steps}")
        node_visited = []
        for _ in range(len(queue)):  # Go only through nodes present in the queue at current timestep
            node = queue.popleft()
            requsted_chunks = list(node.postcondition.difference(node.chunks))
            if len(requsted_chunks) == 0:
                print(node)
                node.state = True
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
                    # match_cand_idx = candidates[0]
                    match_cand_idx = random.choice(candidates) # randomly select a candidate
                    match_cand = mesh.node_map[match_cand_idx]
                    match_cand.sent_num += 1
                    node.recv_num += 1
                    node.add_chunk(chunk)
                    count += 1
                    if node.idx not in node_visited:
                        node_visited.append(node.idx)
                    if match_cand_idx not in node_visited:
                        node_visited.append(match_cand_idx)
                    print(f'send {chunk} from {match_cand.rank} to {node.rank}')

            for idx in node.links:
                neighbor = mesh.node_map[idx]
                if neighbor.state == False:
                    queue.append(neighbor)
            
        for idx in node_visited:
            mesh.node_map[idx].reset_links()
        if count > 0:
            print(f"links transfering at the same time: {count}")
            total_cost += mesh.alpha + mesh.beta
            time_steps += 1
            # mesh.print_cur_chunks()
    # for node in mesh.node_map.values():
    #     print(node.state)
        # print(f"{node.rank}: {sorted(node.postcondition)}")

    return time_steps, total_cost

if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit("Usage: python mesh_search.py M N collective")
    print(Chunk(1,2))
    M, N = int(sys.argv[1]), int(sys.argv[2])
    collective = sys.argv[3]
    alpha = 1           # DGX-2 NVlink is 0.7us
    beta = 10           # DGX-2 NVlink is 8us/MB
    data_size = 1       # send 1MB size chunk
    mesh = Mesh(M, N, alpha, beta)
    print(f"Mesh can send/recv at most {mesh.max_links} links in parallel")
    mesh.init_precondtion(collective)
    mesh.init_postcondition(collective)
    print("cur chunks")
    mesh.print_cur_chunks()
    print("------------- Greedy Search Starts ----------------")
    # cProfile.run("greedy_search(mesh)")
    timesteps, total_cost = greedy_search(mesh)
    print("------------- Greedy Search Ends ------------------")
    print("cur chunks after search")
    mesh.print_cur_chunks()
    
    mesh.print_preconditions()
    mesh.print_postconditions()
    # alpha is 1 (us), beta is 10 (us/MB), about 
    # 100GB = 102400MB
    ring_cost = (M * N - 1) * (alpha + data_size * beta)
    print(f"All-Gather Ring need {M * N - 1} steps, total cost {ring_cost}")
    print(f"Finish All-Gather in {timesteps} time steps, total cost {total_cost}")
