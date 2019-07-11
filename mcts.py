"""
This contains the Node, Edge and MCTS classes, that constitute a Monte Carlo Search Tree.
"""

import numpy as np
from util import logger as lg


class Node:

    def __init__(self, state):
        """
        This initialize a Node in our MCTS Tree. Every Node holds a GameState (BoardSituation), a Playercolor, an ID and
        his edges to the children.

        :param state: The GameState
        """
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.edges = []

    def isLeaf(self):
        """
        This method checks if the edges are 0, so the node is a leaf

        :return: True or False (is leaf)
        """
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge:

    def __init__(self, inNode, outNode, prior, action):
        """
        ...
        :param inNode: The node
        :param outNode:
        :param prior:
        :param action: chessMove
        """
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.state.playerTurn
        self.action = action

        self.stats = {
            'node_visits': 0,
            'node_total_evaluation': 0,
            'node_average_evaluation': 0,
            'action_probability': prior,
        }


class MCTS:

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        lg.logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        currentNode = self.root

        done = 0
        result = 0

        while not currentNode.isLeaf():

            lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)

            maxQU = -99999

            parent_visits = 1
            for action, edge in currentNode.edges:
                parent_visits = parent_visits + edge.stats['node_visits']

            for idx, (action, edge) in enumerate(currentNode.edges):
                if edge not in breadcrumbs:

                    # UCT = Q+U
                    U = self.cpuct * edge.stats['action_probability'] * \
                        np.sqrt((parent_visits) / (1 + edge.stats['node_visits']))
                    Q = edge.stats['node_average_evaluation']

                    lg.logger_mcts.info(
                        'action: %s ... node_visits = %d, action_probability = %f, node_total_evaluation = %f, node_average_evaluation = %f, U = %f, Q+U = %f',
                        action,
                        edge.stats['node_visits'], np.round(edge.stats['action_probability'], 6),
                        np.round(edge.stats['node_total_evaluation'], 6), np.round(Q, 6), np.round(U, 6), np.round(Q + U, 6))

                    if Q + U > maxQU:
                        maxQU = Q + U
                        action_maxQU = action
                        edge_maxQU = edge

            lg.logger_mcts.info('action with highest Q + U...%s', action_maxQU)

            new_state, result, done = currentNode.state.take_action(action_maxQU)
            # whether the game is done and the result from the point of view of the new playerTurn
            # result is 0 if the game is not yet finished.
            currentNode = edge_maxQU.outNode
            breadcrumbs.append(edge_maxQU)

        lg.logger_mcts.info('DONE/Endgame...%d', done)

        return currentNode, result, done, breadcrumbs

    def back_fill(self, leaf, leaf_evaluation, breadcrumbs):
        lg.logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.state.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['node_visits'] = edge.stats['node_visits'] + 1
            edge.stats['node_total_evaluation'] = edge.stats['node_total_evaluation'] + leaf_evaluation * direction
            edge.stats['node_average_evaluation'] = edge.stats['node_total_evaluation'] / edge.stats['node_visits']

            lg.logger_mcts.info('updating edge with leaf_evaluation %f for player %d... N = %d, W = %f, Q = %f', leaf_evaluation * direction, playerTurn, edge.stats['node_visits'], edge.stats['node_total_evaluation'], edge.stats['node_average_evaluation']
                                )

            # edge.outNode.state.render(lg.logger_mcts)

    def add_node(self, node):
        self.tree[node.id] = node


'''
New code 
"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3.
Luke Harold Miles, November 2018, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
"""
from collections import defaultdict
import math


class MCTS:
    "Monte Carlo tree search"

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node"
        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better"
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.uct_select(node)  # descend a layer deeper

    def expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            node2 = node.find_random_child()
            if node2 is None:
                return node.reward()
            node = node2

    def backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in path:
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa
            self.N[node] += 1
            self.Q[node] += reward

    def uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node must be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + \
                self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)


class Node:
    "This can be a checkers or chess or tic-tac-to board state"

    def find_children(self):
        "All possible successors to this board state"
        return set()

    def find_random_child(self):
        "For efficiency in simulation. Returns None if node has no children"
        return None

    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"

        return 0

    def __init__(self, state):
        """
        This initialize a Node in our MCTS Tree. Every Node holds a GameState (BoardSituation), a Playercolor, an ID and
        his edges to the children.

        :param state: The GameState
        """
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.children = []


    def __hash__(self):
        "Nodes must be hashable"
        return 37

    def __eq__(node1, node2):
        "Nodes must be comparable"
return True

'''