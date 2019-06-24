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

            parent_visits = 0
            for action, edge in currentNode.edges:
                parent_visits = parent_visits + edge.stats['node_visits']

            for idx, (action, edge) in enumerate(currentNode.edges):

                # UCT = Q+U
                U = self.cpuct * edge.stats['action_probability'] * \
                    np.sqrt(parent_visits / (1 + edge.stats['node_visits']))
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
