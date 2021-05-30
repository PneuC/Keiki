from heapq import *
from copy import deepcopy
from agents.simple_agents import Agent
from agents.realtime_sim import RealTimeSimulator
from logic.objects.player import Player
from utils.math import Vec2


class StateNode:
    def __init__(self, cnt, path):
        # self.pos = None     # Player's position at this state.
        self.cnt = cnt      # Frame count at this state.
        self.trace = []
        self.path = path    # Path from root.
        self.children = []
        self.evaluated = False
        self.miss = 0
        self.val = 0.0

    def __getitem__(self, item):
        return self.children[item]

    def __lt__(self, other):
        return self.val > other.val

    def set(self, val, trace, miss):
        self.val = val
        self.trace = trace
        self.miss = miss
        self.evaluated = True

    def valid(self):
        return self.evaluated and self.path

    def __getattr__(self, item):
        if item == 'depth':
            return len(self.path)
        elif item == 'pos':
            return self.trace[-1]

class Tree:
    def __init__(self, pos, branches, cnt=0):
        self.nbraches = branches
        self.root = StateNode(cnt, [])
        self.root.set(0.0, [pos], 0)

    def step(self, i, repeat):
        # Reset root
        self.root = self.root[i]
        # Reset path
        stack = [[self.root, 0]]
        while stack:
            top = stack[-1]
            if top[1] < self.nbraches and top[0].children:
                next_node = top[0][top[1]]
                top[1] = top[1] + 1
                stack.append([next_node, 0])
            else:
                stack[-1][0].path.pop(0)
                del stack[-1][0].trace[:repeat]
                stack.pop(-1)

    def get(self, path):
        node = self.root
        for i in path:
            node = self.root[i]
        return node

    def traverse(self, condition=lambda x: True):
        size = 0
        stack = [[self.root, 0]]
        res = []
        while stack:
            top = stack[-1]
            if top[1] < self.nbraches and top[0].children:
                next_node = top[0][top[1]]
                top[1] = top[1] + 1
                stack.append([next_node, 0])
            else:
                if condition(top[0]):
                    res.append(top[0])
                stack.pop()
                size += 1
        # print('tree size: %d' % size)
        return res

    # def show(self):
    #     size = 0
    #     stack = [[self.root, 0]]
    #     while stack:
    #         top = stack[-1]
    #         if top[1] < self.nbraches and top[0].children:
    #             next_node = top[0][top[1]]
    #             top[1] = top[1] + 1
    #             stack.append([next_node, 0])
    #         else:
    #             stack.pop(-1)
    #             size += 1


class AStarAgent(Agent):
    def __init__(self, pos=None, repeat=3, steps=10, max_depth=50, allow_tilt=False):
        super(AStarAgent, self).__init__()
        RealTimeSimulator()
        self.cnt = 0
        self.repeat = repeat
        self.steps = steps
        self.allow_tilt = allow_tilt
        self.branches = 9 if self.allow_tilt else 5
        starting_pos = Player.init_pos.cpy() if pos is None else pos.cpy()
        self.max_depth = max_depth
        self.search_tree = Tree(starting_pos, self.branches, self.cnt)
        self.action = 0
        self.synchronization = 2
        self.min_val = 100.
        self.entropy = 0
        # self.max_depeth = max_depth

    def update(self):
        # if RealTimeSimulator.instance.itab[self.cnt]:
        #     print('itab at %d ')
        if self.synchronization > 0:
            self.synchronization -= 1
            return
        if self.cnt % self.repeat == 0:
            # Construct State Heap
            simulator = RealTimeSimulator.instance
            if not simulator.loaded:
                return
                # raise RuntimeWarning('Simulator not loaded when use')
            max_cnt = simulator.T
            heap = self.search_tree.traverse(lambda x:not x.children)
            heapify(heap)
            # Search
            for _ in range(self.steps):
                if not heap:
                    return
                # Select node to expand
                node_to_expand = heappop(heap)
                while heap and (node_to_expand.children or len(node_to_expand.path) >= self.max_depth
                        or node_to_expand.cnt >= max_cnt - self.repeat):
                    node_to_expand = heappop(heap)
                # Expand the node
                prev_trace, starting_cnt = node_to_expand.trace, node_to_expand.cnt
                for i in range(self.branches):
                    new_path = deepcopy(node_to_expand.path)
                    new_path.append(i)
                    new_node = StateNode(starting_cnt + self.repeat, new_path)
                    dire_vec, _ = Agent.decode_action(i)
                    safety, miss, new_trace = simulator.evaluate(starting_cnt, dire_vec, self.repeat, prev_trace)
                    miss += node_to_expand.miss
                    value = AStarAgent.heuristic(safety=safety, miss=miss, pos=new_trace[-1])
                    new_node.set(value, new_trace, miss)
                    node_to_expand.children.append(new_node)
                    # print(new_node.depth, len(new_node.trace))
                    heappush(heap, new_node)
            # Reset Search Tree
            if heap[0].path[0] != self.action:
                self.entropy += 1
                self.action = heap[0].path[0]
            # print(heap[0].val)
            self.min_val = min(self.min_val, heap[0].val)
            self.search_tree.step(self.action, self.repeat)
        self.cnt += 1

    @staticmethod
    def heuristic(**kwargs):
        value = kwargs['safety'] - kwargs['miss'] * 100 - abs(kwargs['pos'].x) / 100
        # value = kwargs['safety'] - kwargs['miss'] * 100
        return value
#
# if __name__ == '__main__':
#     tree = Tree(Player.init_pos.cpy(), 3)
#     tree.root.children += [StateNode(0, [i]) for i in range(3)]
#     tree.root[1].children += [StateNode(1, [1, i]) for i in range(3)]
#     for node in tree.traverse():
#         print(node.path)
#     print('---------------------------')
#     tree.step(1)
#     for node in tree.traverse():
#         print(node.path)
#     print('---------------------------')
#     # print(tree.root.children)
#     # print(not tree.root.children)
#     for node in tree.traverse(lambda x:not x.children):
#         print(node.path)
#
