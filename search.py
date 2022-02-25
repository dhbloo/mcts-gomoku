import torch
import numpy as np
import math
from collections import defaultdict


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, visit_func, random_choice=False, c_puct=1.0, fpu_reduction=0.2):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.visit_func = visit_func
        self.random_choice = random_choice
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")
        elif node not in self.children:
            raise RuntimeError(f"node {node} has not been expanded")

        if self.N[node] == 1:
            # no child has been visited, select child with best policy prior
            best_child = self._choose_action(node, lambda n: node.policy(n))
        else:
            # select best child with max visits
            best_child = self._choose_action(node, lambda n: self.N[n])

        Q, N = self.Q[best_child], self.N[best_child]
        return best_child, Q / N if N else 0.0, N

    def search(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = 1 - leaf.reward()
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            node = self._puct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded (terminal node)
        self.children[node] = node.expand(self.visit_func)

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _puct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        puct_factor = self.c_puct * math.sqrt(self.N[node])

        explored_policy_sum = sum(
            node.policy(c) for c in self.children[node] if c in self.children)
        init_q_reduction = self.fpu_reduction * math.sqrt(explored_policy_sum)
        init_q = (1 - self.Q[node] / self.N[node]) - init_q_reduction

        def puct(c):
            "Predictor upper confidence bound for trees"
            u = puct_factor * node.policy(c) / (1 + self.N[c])

            if c in self.children:  # child that has been explored for at least once
                q = self.Q[c] / self.N[c]
            else:  # child that has not been explored
                q = init_q
                # q = 0

            return q + u

        return max(self.children[node], key=puct)

    def _choose_action(self, node, key):
        """Choose a (randomly) best action from children of node"""
        if self.random_choice:
            children = [c for c in self.children[node]]
            indices = np.arange(len(children))
            probs = np.array([key(c) for c in children])
            index = np.random.choice(indices, p=probs / probs.sum())
            return children[index]
        else:
            return max(self.children[node], key=key)


class Board():
    DIRECTIONS = [(0, 1), (1, 0), (1, -1), (1, 1)]
    ZOBRIST_TABLE = [[[
        np.random.randint(low=-9223372036854775808, high=9223372036854775807, dtype=np.int64)
        for _ in range(32)
    ] for _ in range(32)] for _ in range(2)]

    def __init__(self, board_width, board_height, fixed_side_input=False, board_to_clone=None):
        if board_to_clone is not None:
            self.board = board_to_clone.board.copy()
            self.move_history = [m for m in board_to_clone.move_history]
            self.side_to_move = board_to_clone.side_to_move
            self.hash = board_to_clone.hash
            self.fixed_side_input = board_to_clone.fixed_side_input
        else:
            self.board = np.zeros((2, board_height, board_width), dtype=np.int8)
            self.move_history = []
            self.side_to_move = 0
            self.hash = Board.ZOBRIST_TABLE[-1][-1][-1]
            self.fixed_side_input = fixed_side_input

    @property
    def ply(self):
        return len(self.move_history)

    @property
    def width(self):
        return self.board.shape[2]

    @property
    def height(self):
        return self.board.shape[1]

    @property
    def last_move(self):
        assert self.ply > 0
        x, y, stm = self.move_history[-1]
        return (x, y)

    def flip_side(self):
        self.side_to_move = 1 - self.side_to_move

    def move(self, x, y):
        assert self.is_legal(x, y), f"Pos ({x},{y}) is not legal!"
        self.board[self.side_to_move, y, x] = 1
        self.hash ^= Board.ZOBRIST_TABLE[self.side_to_move][y][x]
        if self.ply > 0:
            hx, hy, hstm = self.move_history[-1]
            self.hash ^= hash((x, y, hx, hy))
        self.move_history.append((x, y, self.side_to_move))
        self.flip_side()

    def undo(self):
        assert len(self.move_history) > 0, "Can not undo when board is empty!"
        x, y, stm = self.move_history.pop()
        self.board[stm, y, x] = 0
        self.hash ^= Board.ZOBRIST_TABLE[stm][y][x]
        if self.ply > 0:
            hx, hy, hstm = self.move_history[-1]
            self.hash ^= hash((x, y, hx, hy))
        self.side_to_move = stm

    def is_inboard(self, x, y):
        return x >= 0 and x < self.board.shape[2] and y >= 0 and y < self.board.shape[1]

    def is_legal(self, x, y):
        return self.is_inboard(x, y) and self.board[0, y, x] == 0 and self.board[1, y, x] == 0

    def is_draw(self):
        "Returns true if the board has been fulfilled"
        return self.ply == self.width * self.height

    def has_five(self):
        "Returns true if last move on board has made a five connection"
        if self.ply == 0:
            return False

        x, y, stm = self.move_history[-1]
        for dx, dy in Board.DIRECTIONS:
            count = 1
            for i in range(1, 5):
                xi, yi = x - i * dx, y - i * dy
                if self.is_inboard(xi, yi) and self.board[stm, yi, xi]:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                xi, yi = x + i * dx, y + i * dy
                if self.is_inboard(xi, yi) and self.board[stm, yi, xi]:
                    count += 1
                else:
                    break
            if count >= 5:
                return True

        return False

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.is_draw() or self.has_five()

    def expand(self, visit_func):
        "Expand a node with visit function."
        if self.is_draw():
            self.value = 0.5
            return set()
        elif self.has_five():
            self.value = 0
            return set()
        else:
            value, policy_prior = visit_func(self.get_data())
            self.value = value.item()
            self.policy_prior = dict()

            children = set()
            # find all possible successors of this board state
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[0, y, x] == 0 and self.board[1, y, x] == 0:
                        child = Board(0, 0, board_to_clone=self)
                        child.move(x, y)
                        self.policy_prior[child] = policy_prior[y, x].item()
                        children.add(child)
            return children

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return self.value

    def policy(self, child):
        "Returns policy prior for a child node"
        return self.policy_prior.get(child, 0)

    def get_data(self):
        "Returns data for neural-net in pytorch format"
        if not self.fixed_side_input and self.side_to_move == 1:
            board_input = np.flip(self.board, axis=0).copy()
        else:
            board_input = self.board

        return {
            'board_size': np.array(self.board.shape[1:], dtype=np.int8),
            'board_input': board_input,
            'stm_input': -1 if self.side_to_move == 0 else 1
        }

    def __hash__(self):
        "Nodes must be hashable"
        return self.hash.item()

    def __eq__(node1, node2):
        "Nodes must be comparable"
        if node1.hash != node2.hash:
            return False
        else:
            return (np.array_equal(node1.board, node2.board)
                    and node1.move_history == node2.move_history)

    def __str__(self):
        "Pretty print board"
        s = '   '
        for x in range(self.board.shape[2]):
            s += chr(x + ord('A')) + ' '
        s += '\n'
        for y in range(self.board.shape[1]):
            s += f'{y + 1:2d} '
            for x in range(self.board.shape[2]):
                if self.board[0, y, x]:
                    s += 'X '
                elif self.board[1, y, x]:
                    s += 'O '
                else:
                    s += '. '
            s += '\n'
        return s