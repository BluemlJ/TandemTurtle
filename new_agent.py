"""
Contains class Simple_Agent #TODO move to agent.py. delete old agent.py
Objects of Simple_Agent class are simple Bughouse players that do not use a
trained model. The evaluation is done by a hard-coded evaluation function as defined in eval.py
"""
import time

import numpy as np
import random
import mcts
from game import input_representation, output_representation
from util import logger as lg
import config
from tensorflow.python.keras.backend import set_session


class Agent():
    ##########
    # param:
    # name - agent names
    # state size
    # action size
    # number of MCTS simulations
    # cpuct - exploration coefficient for uct
    # model - the neural net. Not used in simple agent, but kept here for the purpose of later extension
    # interface - function to be called for xboard output commands
    ##########
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model, interface, model_extra, timed_match=False, seconds_per_move=5):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.timed_match = timed_match
        self.seconds_per_move = seconds_per_move

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model  # use later

        # mcts saves tree info and statistics.
        self.root = None

        # save model_extra for graph and session for model.predict is: [graph, sess]
        self.model_extra = model_extra

        self.interface = interface

        # to plot value_head and policy_head loss later
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def suggest_move(self, higher_noise=True):
        """Used for playing a single game.

        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        """
        start = time.time()
        # expand root if not expanded yet
        if not self.root.is_expanded:
            prob, val = self.get_preds([self.root.state])
            self.root.incorporate_results(prob[0], val[0], self.root)

        if self.timed_match:
            while time.time() - start < self.seconds_per_move:
                self.tree_search()
        else:
            current_readouts = self.root.N
            while self.root.N < current_readouts + self.MCTSsimulations:
                self.tree_search()

        return self.pick_move(higher_noise)  # TODO reimplement setting of high noise

    def play_move(self, move, on_partner_board):
        """Notable side effects:
          - finalizes the probability distribution according to
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        """
        if not on_partner_board:
            move.board_id = self.root.state.board.board_id
            fmove = output_representation.move_to_policy_idx(move, is_white_to_move=self.root.state.board.turn)
            self.root = self.root.maybe_add_child(fmove)
            del self.root.parent.children
        else:
            move.board_id = self.root.state.partner_board.board_id
            new_state, _, _ = self.root.state.take_action(move)
            self.build_mcts(new_state)

        self.state = self.root.state

        return True  # GTP requires positive result.

    def pick_move(self, higher_noise):
        """Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max."""
        if not higher_noise:
            fcoord = self.root.best_child()
        else:
            cdf = self.root.children_as_pi(squash=True).cumsum()
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        move = output_representation.policy_idx_to_move(fcoord, self.root.state.board.turn, self.root.state.board.board_id)
        return move

    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = min(config.PARALLEL_READOUTS, self.MCTSsimulations)
        leaves = []
        failsafe = 0
        while len(leaves) < parallel_readouts and failsafe < parallel_readouts * 2 and failsafe < len(self.root.state.allowedActions):
            failsafe += 1
            leaf = self.root.select_leaf()

            # if game is over, override the value estimate with the true score
            if leaf.is_done():
                value = 1 if leaf.state.value[0] > 0 else -1
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.get_preds([leaf.state for leaf in leaves])
            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)
        return leaves

    def get_preds(self, states):
        # predict the leaf
        inputs1 = []
        inputs2 = []
        for state in states:
            board = state.board
            partner_board = state.partner_board

            x1 = input_representation.board_to_planes(board)
            x1 = np.expand_dims(x1, axis=0)
            x2 = input_representation.board_to_planes(partner_board)
            x2 = np.expand_dims(x2, axis=0)
            inputs1.append(x1)
            inputs2.append(x2)

        inputs = {"input_1": np.concatenate(inputs1), "input_2": np.concatenate(inputs2)}
        with self.model_extra[0].as_default():
            set_session(self.model_extra[1])
            predictions = self.model.predict(inputs)

        # value head should be one value to say how good my state is
        value_head = predictions[0]
        # policy head gives a 2272 big vector with prob for each state
        policy_head = predictions[1]

        return policy_head, value_head

    def act_nn(self, state, higher_noise, deterministic=False):
        """
        Run without simulations or mcts, get move probs from NN and sample from this distr
        :param state: Current state
        :param higher_noise: not used
        :param deterministic: If move is sampled or chosen with argmax
        :return: best single move
        """

        value_head, move_probabilities, allowed_action_idxs, allowed_actions = self.get_preds(state)
        move_probabilities = move_probabilities[allowed_action_idxs]
        print("move probs: ", move_probabilities)

        if deterministic:
            best_move_idx = np.argmax(move_probabilities)
        else:
            best_move_idx = np.random.choice(len(move_probabilities), p=move_probabilities)

        best_move = allowed_actions[best_move_idx]

        return best_move

    def act_random(self, state):
        """
        Returns random allowed action given current state
        :param state:
        :return:
        """
        allowed_actions = state.allowedActions
        rand_act = np.random.choice(len(allowed_actions))

        return allowed_actions[rand_act]

    def build_mcts(self, state):

        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mcts.MCTSNode(state)
        self.result = 0
        self.result_string = None
        self.comments = []
