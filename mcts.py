"""
This contains the Node, Edge and MCTS classes, that constitute a Monte Carlo Search Tree.
"""
import collections

import math
import numpy as np
from util import logger as lg
from game import constants as game_constants, output_representation
import config as cf


class DummyNode(object):
    """A fake node of a MCTS search tree.

    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class MCTSNode(object):
    """A node of a MCTS search tree.

    A node knows how to compute the action scores of all of its children,
    so that a decision can be made about which move to explore next. Upon
    selecting a move, the children dictionary is updated with a new node.

    position: A go.Position instance
    fmove: A move (coordinate) that led to this position, a a flattened coord
            (raw number between 0-N, with None a pass)
    parent: A parent MCTSNode.
    """

    def __init__(self, state, fmove=None, parent=None):
        if parent is None:
            parent = DummyNode()

        n = game_constants.NB_LABELS
        self.parent = parent
        self.fmove = fmove  # move that led to this position, as flattened coords
        self.state = state
        self.is_expanded = False
        self.losses_applied = 0  # number of virtual losses on this node

        allowedActions_idxs = [output_representation.move_to_policy_idx(move, is_white_to_move=self.state.board.turn)
                               for move in self.state.allowedActions]
        legal_moves = np.zeros(n)
        legal_moves[allowedActions_idxs] = 1

        self.illegal_moves = 1 - legal_moves

        # using child_() allows vectorized computation of action score.
        self.child_N = np.zeros(n, dtype=np.float32)
        self.child_W = np.zeros(n, dtype=np.float32)
        # save a copy of the original prior before it gets mutated by d-noise.
        self.original_prior = np.zeros(n, dtype=np.float32)
        self.child_prior = np.zeros(n, dtype=np.float32)
        self.children = {}  # map of flattened moves to resulting MCTSNode

    # def __repr__(self):
    #     return "<MCTSNode move=%s, N=%s, to_play=%s>" % (
    #         self.position.recent[-1:], self.N, self.state.playerTurn)

    @property
    def child_action_score(self):
        return (self.child_Q * self.state.playerTurn +
                self.child_U - 1000 * self.illegal_moves)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):

        return ((2.0 * (
                math.log((1.0 + self.N + cf.CPUCT_BASE) / cf.CPUCT_BASE) + cf.CPUCT))
                * math.sqrt(max(1, self.N - 1)) * self.child_prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def N(self):
        return self.parent.child_N[self.fmove]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.fmove] = value

    @property
    def W(self):
        return self.parent.child_W[self.fmove]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.fmove] = value

    @property
    def Q_perspective(self):
        "Return value of position, from perspective of player to play."
        return self.Q * self.state.playerTurn

    def select_leaf(self):
        current = self

        while True:
            # if a node has never been evaluated, we have no basis to select a child.
            if not current.is_expanded:
                break

            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, fcoord):
        """ Adds child node for fcoord if it doesn't already exist, and returns it. """
        if fcoord not in self.children:
            move = output_representation.policy_idx_to_move(fcoord, self.state.board.turn, self.state.board.board_id)
            new_position, value, done = self.state.take_action(move)

            self.children[fcoord] = MCTSNode(new_position, fmove=fcoord, parent=self)
        return self.children[fcoord]

    def add_virtual_loss(self, up_to):
        """Propagate a virtual loss up to the root node.

        Args:
            up_to: The node to propagate until. (Keep track of this! You'll
                need it to reverse the virtual loss later.)
        """
        self.losses_applied += 1
        # This is a "win" for the current node; hence a loss for its parent node
        # who will be deciding whether to investigate this node again.
        loss = self.state.playerTurn
        self.W += loss
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.losses_applied -= 1
        revert = -1 * self.state.playerTurn
        self.W += revert
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        assert move_probabilities.shape == (output_representation.NB_LABELS,)
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        assert not self.state.isEndGame

        # If a node was picked multiple times (despite vlosses), we shouldn't
        # expand it more than once.
        if self.is_expanded:
            return
        self.is_expanded = True

        # Zero out illegal moves.
        move_probs = move_probabilities * (1 - self.illegal_moves)
        scale = sum(move_probs)
        if scale > 0:
            # Re-normalize move_probabilities.
            move_probs *= 1 / scale

        self.original_prior = self.child_prior = move_probs
        # initialize child Q as current node's value, to prevent dynamics where
        # if B is winning, then B will only ever explore 1 move, because the Q
        # estimation will be so much larger than the 0 of the other moves.
        #
        # Conversely, if W is winning, then B will explore all 362 moves before
        # continuing to explore the most favorable move. This is a waste of search.
        #
        # The value seeded here acts as a prior, and gets averaged into Q calculations.
        self.child_W = np.ones(output_representation.NB_LABELS, dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """Propagates a value estimation up to the root node.

        Args:
            value: the value to be propagated (1 = black wins, -1 = white wins)
            up_to: the node to propagate until.
        """
        self.N += 1
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        """True if the last two moves were Pass or if the position is at a move
        greater than the max depth."""
        return self.state.isEndGame

    def inject_noise(self):
        epsilon = 1e-5
        legal_moves = (1 - self.illegal_moves) + epsilon
        a = legal_moves * ([cf.DIRICHLET_ALPHA] * (output_representation.NB_LABELS))
        dirichlet = np.random.dirichlet(a)
        self.child_prior = (self.child_prior * (1 - cf.DIRICHLET_WEIGHT) +
                            dirichlet * cf.DIRICHLET_WEIGHT)

    def children_as_pi(self, squash=False):
        """Returns the child visit counts as a probability distribution, pi
        If squash is true, exponentiate the probabilities by a temperature
        slightly larger than unity to encourage diversity in early play and
        hopefully to move away from 3-3s
        """
        probs = self.child_N
        if squash:
            probs = probs ** (1 - cf.TEMPERATURE)
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return probs
        return probs / np.sum(probs)

    def best_child(self):
        # Sort by child_N tie break with action score.
        return np.argmax(self.child_N + self.child_action_score / 10000)

    def most_visited_path_nodes(self):
        node = self
        output = []
        while node.children:
            node = node.children.get(node.best_child())
            assert node is not None
            output.append(node)
        return output

    def most_visited_path(self):
        output = []
        node = self
        for node in self.most_visited_path_nodes():
            output.append("%s (%d) ==> " % (
                output_representation.move_to_policy_idx(node.fmove, node.state.board.turn), node.N))

        output.append("Q: {:.5f}\n".format(node.Q))
        return ''.join(output)

    def rank_children(self):
        ranked_children = list(range(game_constants.NB_LABELS))
        ranked_children.sort(key=lambda i: (
            self.child_N[i], self.child_action_score[i]), reverse=True)
        return ranked_children

    def describe(self):
        ranked_children = self.rank_children()
        soft_n = self.child_N / max(1, sum(self.child_N))
        prior = self.child_prior
        p_delta = soft_n - prior
        p_rel = np.divide(p_delta, prior, out=np.zeros_like(
            p_delta), where=prior != 0)
        # Dump out some statistics
        output = []
        output.append("{q:.4f}\n".format(q=self.Q))
        output.append(self.most_visited_path())
        output.append(
            "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  p-rel")
        for i in ranked_children[:15]:
            if self.child_N[i] == 0:
                break
            output.append("\n{!s:4} : {: .3f} {: .3f} {:.3f} {:.3f} {:.3f} {:5d} {:.4f} {: .5f} {: .2f}".format(
                output_representation.move_to_policy_idx(i),
                self.child_action_score[i],
                self.child_Q[i],
                self.child_U[i],
                self.child_prior[i],
                self.original_prior[i],
                int(self.child_N[i]),
                soft_n[i],
                p_delta[i],
                p_rel[i]))
        return ''.join(output)


# TODO old code. Delete
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
                        np.round(edge.stats['node_total_evaluation'], 6), np.round(Q, 6), np.round(U, 6),
                        np.round(Q + U, 6))

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

            lg.logger_mcts.info('updating edge with leaf_evaluation %f for player %d... N = %d, W = %f, Q = %f',
                                leaf_evaluation * direction, playerTurn, edge.stats['node_visits'],
                                edge.stats['node_total_evaluation'], edge.stats['node_average_evaluation']
                                )

            # edge.outNode.state.render(lg.logger_mcts)

    def add_node(self, node):
        self.tree[node.id] = node
