"""
Contains class Simple_Agent
Objects of Simple_Agent class are simple Bughouse players that do not use a
trained model. The evaluation is done by a hard-coded evaluation function as defined in eval.py
"""
import numpy as np
import random
import mcts as mc
import eval
from game import input_representation, output_representation
from util import logger as lg


class Simple_Agent():
    ##########
    # param:
    # name - agent name
    # state size
    # action size
    # number of MCTS simulations
    # cpuct - exploration coefficient for uct
    # model - the neural net. Not used in simple agent, but kept here for the purpose of later extension
    # interface - function to be called for xboard output commands
    ##########
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model, interface):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model  # use later

        # mcts saves tree info and statistics.
        self.mcts = None

        self.interface = interface

        # to plot value and policy loss later
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    ##########
    # do one simulation of a game and evaluate the outcome. Update the MC search tree in the progress.
    ##########
    def simulate(self):

        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(lg.logger_mcts)  # log game state
        lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        # MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()
        leaf.state.render(lg.logger_mcts)

        # EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs)

        # BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.back_fill(leaf, value, breadcrumbs)

    ####
    # act - run simulations updating the MC-search-tree. Then pick an action.
    # param:
    # state - the game state
    # higher_noise: 1 (in the beginning, when the moves are not yet deterministic)
    #       or 0 (after some time, when the simple_agent starts playing deterministically.
    # returns:
    # action - the chosen action,
    # edge_visited_rates - how often (relatively) the actions/edges were visited by mcts
    # value - (?)
    # nn_value - zero in this simple case
    # .

    def act(self, state, higher_noise):
        # go to mcts node that corresponds to state or build new mcts.
        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state)  # so that it can use previous simulations

        # run the simulation
        for sim in range(self.MCTSsimulations):  # TODO (later) use fixed time instead of fixed nr of simulations
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('***************************')
            self.simulate()  # updates MCTS

        # get action values. edge_visited_rates are, how frequently an edge/action is visited
        edge_visited_rates, win_rates = self.get_action_values()

        # pick the action where visited_rate is max.
        action, win_rate = self.choose_action(edge_visited_rates, win_rates, higher_noise)

        nextState, _, _ = state.take_action(action)  # only needed for nn_value
        # ---> TODO implement get preds = NN_value = -self.get_preds(nextState)[0]
        # TODO implement NN value!! only temporary
        NN_value = self.get_preds(nextState)[0]

        lg.logger_mcts.info('EDGE_VISITED_RATE...%s', edge_visited_rates)
        lg.logger_mcts.info('CHOSEN ACTION...%s', action)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)
        return (action, edge_visited_rates, win_rate, NN_value)

    def get_preds(self, state):
        # predict the leaf
        board = state.board
        partner_board = state.partner_board

        x1 = input_representation.board_to_planes(board)
        x1 = np.expand_dims(x1, axis=0)
        x2 = input_representation.board_to_planes(partner_board)
        x2 = np.expand_dims(x2, axis=0)

        input_to_model = [x1, x2]

        predictions = self.model.predict(input_to_model)
        # value head should be one value to say how good my state is
        value_head = predictions[0]
        # policy head gives a 2272 big vector with prob for each state
        policy_head = predictions[1][0]

        allowed_action_idxs = [output_representation.move_to_policy_idx
                               (move, is_white_to_move=board.turn) for move in state.allowedActions]

        mask = np.ones(policy_head.shape, dtype=bool)
        mask[allowed_action_idxs] = False
        policy_head[mask] = -100

        odds = np.exp(policy_head)
        probs = odds / np.sum(odds)

        allowed_actions = [output_representation.policy_idx_to_move
                           (idx, is_white_to_move=board.turn, board_id=board.board_id) for idx in allowed_action_idxs]

        return value_head, probs, allowed_action_idxs, allowed_actions

    ####
    # evaluate_leaf: .
    def evaluate_leaf(self, leaf, eval_value, done, breadcrumbs):  # TODO (later): delete breadcrumbs, its not used, is it?

        lg.logger_mcts.info('------EVALUATING LEAF------')
        if done == 0:

            value, probs, allowed_action_idxs, allowed_actions = self.get_preds(leaf.state)
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            probs = probs[allowed_action_idxs]

            # ---- TODO delete above and use get_preds

            for idx, action in enumerate(allowed_actions):
                newState, _, _ = leaf.state.take_action(action)
                if newState.id not in self.mcts.tree:
                    node = mc.Node(newState)
                    self.mcts.add_node(node)
                    lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
                else:
                    node = self.mcts.tree[newState.id]
                    lg.logger_mcts.info('existing node...%s...', node.id)

                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))

        else:  # after evaluation is done (done ==1)
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, eval_value)

        return (eval_value, breadcrumbs)

    def get_action_values(self):

        edges = self.mcts.root.edges
        edge_visited_rates = {}
        win_rates = {}
        rates_total = 0

        for action, edge in edges:
            # Todo will only take first argmax, but several ones in actions
            edge_visited_rate = edge.stats['N']
            rates_total += edge_visited_rate
            edge_visited_rates[action] = edge_visited_rate
            win_rates[action] = edge.stats['Q']

        # prevent division by zero error. In case there are no edges visited the actions/edges can be chosen arbitrarily.
        if rates_total == 0:
            rates_total = 1

        for key, value in edge_visited_rates.items():
            # normalize edge_visited_rate to sum up to 1 (probability distribution)
            edge_visited_rates[key] = value / (rates_total * 1.0)

        return edge_visited_rates, win_rates

    ####
    # choose_action: pick the action where the visited rate is max. In the first few rounds:
    # choose an action with higher probability, where the visited rate is higher.
    # param: edges_visited_rates, values (map actions to their values), higher_noise (in the first few rounds the noise is higher)
    # return: action and its corresponding value
    ####

    def choose_action(self, edges_visited_rates, win_rates, higher_noise):
        # invert dictionary - swap key and value. Now visited rate is in front, action is second
        # [(action1, visited_rate1),...] -> [(visited_rate1, action1), ...]
        # TODO remove square brackets for performance
        inverse = [(value, key) for key, value in edges_visited_rates.items()]
        visited_rates = [value for value, key in inverse]  # without the corresponding actions/edges

        if higher_noise == 0:
            action_idx = [i for i, vr in enumerate(visited_rates) if vr == max(visited_rates)]
            actions = [inverse[idx][1] for idx in action_idx]
            action = random.choice(actions)
        else:
            value_idx_arr = np.random.multinomial(1, visited_rates)
            value_idx = np.where(value_idx_arr == 1)[0][0]
            action = inverse[value_idx][1]

        win_rate = win_rates[action]

        # return action
        return action, win_rate

    # TODO reimplement replay and predict

    def build_mcts(self, state):

        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):

        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
