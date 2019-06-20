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
    """
    ...
    """

    def __init__(self, inNode, outNode, prior, action):
        """
        ...
        :param inNode:
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
            'N': 0,
            'W': 0,
            'Q': 0,
            # 'P': prior, not needed yet (only for NN Approach, for simple approach P = 1
            'P': 1,
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
        value = 0

        while not currentNode.isLeaf():

            lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)

            maxQU = -99999

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):

                U = self.cpuct * \
                    edge.stats['P'] * \
                    np.sqrt(Nb / (1 + edge.stats['N']))

                Q = edge.stats['Q']

                lg.logger_mcts.info(
                    'action: %s ... N = %d, P = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f',
                    action,
                    edge.stats['N'],
                    np.round(
                        edge.stats['P'],
                        6),
                    (edge.stats['P']),
                    np.round(
                        edge.stats['W'],
                        6),
                    np.round(
                        Q,
                        6),
                    np.round(
                        U,
                        6),
                    np.round(
                        Q + U,
                        6))

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            lg.logger_mcts.info('action with highest Q + U...%s', simulationAction)

            newState, value, done = currentNode.state.take_action(simulationAction)
            # the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        lg.logger_mcts.info('DONE...%d', done)

        return currentNode, value, done, breadcrumbs

    def back_fill(self, leaf, value, breadcrumbs):
        lg.logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.state.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f', value * direction, playerTurn, edge.stats['N'], edge.stats['W'], edge.stats['Q']
                                )

            # edge.outNode.state.render(lg.logger_mcts)

    def add_node(self, node):
        self.tree[node.id] = node
