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
    # pi - the priorities (?) of the different actions,
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

        # get action values. pi is a probability distribution over the visited nodes.
        pi, values = self.get_action_values(1)

        # pick the action where pi is max.
        action, value = self.choose_action(pi, values, higher_noise)

        nextState, _, _ = state.take_action(action)  # only needed for nn_value
        # ---> TODO implement get preds = NN_value = -self.get_preds(nextState)[0]
        # TODO implement NN value!! only temporary
        NN_value = self.get_preds(nextState)[0]

        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%s', action)
        # lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

        # return (action, pi, value, nn_value)
        return (action, pi, value, NN_value)

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
        print("predicted")
        # value head should be one value to say how good my state is
        value_head = predictions[0]
        # policy head gives a 2272 big vector with prob for each state
        policy_head = predictions[1]

        print(value_head)
        print(policy_head)

        allowedActions = [output_representation.move_to_policy
                          (move, is_white_to_move=board.turn) for move in state.allowedActions]

        print(np.array(allowedActions).shape)
        # shape 20, 2272
        print("po shape :", policy_head.shape)
        exit()
        mask = np.ones(policy_head.shape, dtype=bool)
        mask[allowedActions] = False
        policy_head[mask] = -100

        odds = np.exp(policy_head)
        probs = odds / np.sum(odds)

        return value_head, probs, allowedActions

    ####
    # evaluate_leaf: .
    def evaluate_leaf(self, leaf, eval_value, done, breadcrumbs):  # TODO (later): delete breadcrumbs, its not used, is it?

        lg.logger_mcts.info('------EVALUATING LEAF------')
        if done == 0:

            value, probs, allowedActions = self.get_preds(leaf.state)
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            probs = probs[allowedActions]

            # ---- TODO delete above and use get_preds

            for idx, action in enumerate(allowedActions):
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

    def get_action_values(self, tau):  # TODO: get rid of this tau or give it a better name after we know what it does

        edges = self.mcts.root.edges
        # old:
        # pi = np.zeros(self.action_size, dtype=np.integer)
        # values = np.zeros(self.action_size, dtype=np.float32)
        pi = {}
        values = {}
        pi_total = 0

        for action, edge in edges:
            # Todo will only take first argmax, but several ones in actions
            pi_val = pow(edge.stats['N'], 1 / tau)  # TODO (later) why not use p[action = edge.stats['N'] directly?
            pi_total += pi_val
            pi[action] = pi_val
            values[action] = edge.stats['Q']

        if pi_total == 0:
            pi_total = 1

        for key, value in pi.items():
            # normalize pi to sum up to 1 (probability distribution)
            pi[key] = value / (pi_total * 1.0)

        return pi, values

    ####
    # choose_action: pick the action where pi is max.
    # param: pi, values (map actions to their values), higher_noise (in the first few rounds the noise is higher)
    # return: action and its corresponding value
    ####

    def choose_action(self, pi, values, higher_noise):
        inverse = [(value, key) for key, value in pi.items()]
        pi_values = [value for value, key in inverse]

        if higher_noise == 0:
            action_idx = [i for i, x in enumerate(pi_values) if x == max(pi_values)]
            actions = [inverse[idx][1] for idx in action_idx]
            action = random.choice(actions)
        else:
            value_idx_arr = np.random.multinomial(1, pi_values)
            value_idx = np.where(value_idx_arr == 1)[0][0]
            action = inverse[value_idx][1]

        value = values[action]

        # return action
        return action, value

    # TODO reimplement replay and predict

    def build_mcts(self, state):

        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):

        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
