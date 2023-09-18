import argparse
import sys
import random
import numpy
from dataclasses import dataclass
from collections import deque
import cProfile
from enum import Enum

class Collective(Enum):
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_REDUCE = 3
    
@dataclass
class Chunk:
    chunk_id: int
    rank_id: int

    def __str__(self):
        return f"({self.chunk_id}, {self.rank_id})"
    
class MeshNode(object):
    def __init__(self, idx, rank, links, alpha, betas, coll, data=None):
        self.idx = idx
        self.rank = rank
        self.links = links # link node idx in the mesh
        self.alpha = alpha # link latency
        self.betas = betas # double side link bandwidth
        self.postcondition = set()  # set of Chunks
        self.chunks = set()         # set of Chunks
        self.data_size = 1024 # 1KB
        self.state = False
        self.sent_num = 0
        self.recv_num = 0
        self.collective = coll

    def add_chunk(self, chunk):
        self.chunks.add(chunk)

    def del_chunk(self, chunk):
        self.chunks.remove(chunk)

    def is_finished(self):
        if self.collective == Collective.ALL_GATHER:
            return self.chunks == self.postcondition
        elif self.collective == Collective.REDUCE_SCATTER:
            return (self.chunks & self.postcondition) == self.postcondition
    
    def reset_links(self):
        self.sent_num = 0
        self.recv_num = 0
    
    def __str__(self):
        return f"Node {self.idx}, chunks {self.chunks}, links {self.links}"


class Mesh(object):
    def __init__ (self, m: int, n: int, alpha: float, beta: float, coll: Collective):
        self.row_len = m
        self.col_len = n
        self.rank_num = m * n
        self.alpha = alpha      # us
        self.beta = beta        # us/MB
        self.collective = coll
        self.node_map = self.create_node_map(m, n)
        self.max_links = 2 * (m * (n - 1) + (m - 1) * n)
        # self.precondition = set()
        # self.postcondition = set()
        self.finished = 0
        self.chunk_sum = sum([x + 1 for x in range(m * n)])

    def __str__(self):
        mesh_str = "[" + "1-" * (self.col_len - 1) + "1]\n"
        mesh_str *= self.row_len
        return mesh_str

    # ([Chunk c, NPU n) pair
    def init_precondtion(self, collective: str):
        if collective == Collective.ALL_GATHER:
            for i, node in enumerate(self.node_map.values()):
                node.chunks.add(i + 1)

        elif collective == Collective.REDUCE_SCATTER:
            for i, node in enumerate(self.node_map.values()):
                for j in range(self.rank_num):
                    node.chunks.add((i + 1, j + 1))

    def init_postcondition(self, collective: str):
        if collective == Collective.ALL_GATHER:
            for rank_id, node in enumerate(self.node_map.values()):
                for chunk_id in range(1, self.rank_num + 1):
                    node.postcondition.add(chunk_id)

        elif collective == Collective.REDUCE_SCATTER:
            for i, node in enumerate(self.node_map.values()):
                for j in range(self.rank_num):
                    node.postcondition.add((j + 1, i + 1))
                
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
                node_map[idx] = MeshNode(idx, rank, links, self.alpha, self.beta, self.collective)
                rank += 1                
        return node_map

    def is_finished(self):
        for node in self.node_map.values():
            if node.state == False:
                return False
        return True

    def print_cur_chunks(self):
        for node in self.node_map.values():
            print(f"{node.rank}: {sorted(node.chunks)} {node.state}")
    
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

def greedy_search(mesh: Mesh, coll: Collective):

    queue = deque(mesh.node_map.values())
    time_steps = 0
    total_cost = 0
    count = 0
    # in each time step each directed link can be only used once
    while queue: 
        # print(f"mesh: {mesh.is_finished()}")
        count = 0
        print(f"Time Step {time_steps}")
        node_visited = []
        for _ in range(len(queue)):  # Go only through nodes present in the queue at current timestep
            node = queue.popleft()
            if coll == Collective.ALL_GATHER:
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
                        if node.idx not in node_visited:
                            node_visited.append(node.idx)
                        if match_cand_idx not in node_visited:
                            node_visited.append(match_cand_idx)

                        print(f'send {chunk} from {match_cand.rank} to {node.rank}')
            
            # elif coll == Collective.REDUCE_SCATTER:
                # for idx in node.links:
                #     neighbor = mesh.node_map[idx]
                #     neighbor_chunks = node.chunks.difference(neighbor.chunks)
                #     post_chunks = node.postcondition - node.chunks
                #     chunks = neighbor_chunks | post_chunks
                #     if not chunks:
                #         break
                #     # print(chunks)
                #     for chunk in post_chunks:
                #         if node.recv_num < len(node.links) and neighbor.sent_num < len(neighbor.links) \
                #             and chunk not in node.chunks and chunk in neighbor.chunks:
                #             node.add_chunk(chunk)
                #             node.recv_num += 1
                #             neighbor.sent_num += 1
                #             count += 1
                #             print(f'send {chunk} from {neighbor.rank} to {node.rank}') 
                    
                #     for chunk in neighbor_chunks:
                #         if neighbor.sent_num < len(neighbor.links) and node.sent_num < len(node.links):
                #             neighbor.add_chunk(chunk)
                #             neighbor.recv_num += 1
                #             node.sent_num += 1
                #             count += 1
                #             print(f'send {chunk} from {node.rank} to {neighbor.rank}')
                    
                #     if neighbor.idx not in node_visited:
                #         node_visited.append(neighbor.idx)
                # if node.idx not in node_visited:
                #     node_visited.append(node.idx)

            if node.is_finished():
                node.state = True
                node.chunks = node.postcondition & node.chunks

            if node.state == False:
                queue.append(node)

            for idx in node.links:
                neighbor = mesh.node_map[idx]
                if neighbor.state == False and neighbor not in queue:
                    queue.append(neighbor)
        
        for idx in node_visited:
            mesh.node_map[idx].reset_links()
        if count > 0:
            print(f"links transfering at the same time: {count}")
            total_cost += mesh.alpha + mesh.beta
            time_steps += 1

    return time_steps, total_cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--coll", type=int, required=True, default=1)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--col", type=int, default=3)

    args = parser.parse_args()
    M, N = args.row, args.col
    collective = Collective(args.coll)
    alpha = 1           # DGX-2 NVlink is 0.7us
    beta = 10           # DGX-2 NVlink is 8us/MB
    data_size = 1       # send 1MB size chunk
    mesh = Mesh(M, N, alpha, beta, collective)
    print(f"Mesh can send/recv at most {mesh.max_links} links in parallel")
    mesh.init_precondtion(collective)
    mesh.init_postcondition(collective)
    mesh.print_preconditions()
    mesh.print_postconditions()
    print("-----------------cur chunks------------------------")
    mesh.print_cur_chunks()
    print("------------- Greedy Search Starts ----------------")
    # cProfile.run("greedy_search(mesh, collective)")
    timesteps, total_cost = 0, 0
    timesteps, total_cost = greedy_search(mesh, collective)
    print("------------- Greedy Search Ends ------------------")
    print("cur chunks after search")
    mesh.clear_cur_chunks()
    mesh.print_cur_chunks()
    mesh.print_postconditions()
    
    # alpha is 1 (us), beta is 10 (us/MB), about 
    # 100GB = 102400MB
    ring_cost = (M * N - 1) * (alpha + data_size * beta)
    print(f"{collective} Ring need {M * N - 1} steps, total cost {ring_cost}")
    print(f"Finish {collective} in {timesteps} time steps, total cost {total_cost}")
