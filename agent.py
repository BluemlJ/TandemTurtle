"""
Contains class Simple_Agent
Objects of Simple_Agent class are simple Bughouse players that do not use a
trained model. The evaluation is done by a hard-coded evaluation function as defined in eval.py
"""
import numpy as np
import random
import mcts as mc
from game import input_representation, output_representation
from util import logger as lg
import config


class Agent():
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
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model, interface, graph):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model  # use later

        # mcts saves tree info and statistics.
        self.mcts = None

        # save graph for model.predict
        self.graph = graph

        self.interface = interface

        # to plot value_head and policy_head loss later
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

        # MOVE TO THE LEAF NODE
        leaf, result, done, breadcrumbs = self.mcts.move_to_leaf()
        # start logger.
        leaf.state.render(lg.logger_mcts)

        # EVALUATE THE LEAF NODE
        leaf_evaluation = self.expand_and_evaluate_leaf(leaf, result, done)

        # BACKFILL THE EVALUATION THROUGH THE TREE
        self.mcts.back_fill(leaf, leaf_evaluation, breadcrumbs)

    ####
    # act - run simulations updating the MC-search-tree. Then pick an action.
    # param:
    # state - the game state
    # higher_noise: 1 (in the beginning, when the moves are not yet deterministic)
    #       or 0 (after some time, when the simple_agent starts playing deterministically.
    # returns:
    # action - the chosen action,
    # edge_visited_rates - how often (relatively) the actions/edges were visited by mcts
    #
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

        edge_visited_rates, node_average_evaluations = self.get_statistics_of_root_edges()

        # pick the action where visited_rate is max.
        action, best_average_evaluation = self.choose_action(edge_visited_rates, node_average_evaluations, higher_noise)

        next_state, _, _ = state.take_action(action)
        next_state_evaluation = self.get_preds(next_state)[0]

        lg.logger_mcts.info('EDGE_VISITED_RATE...%s', edge_visited_rates)
        lg.logger_mcts.info('CHOSEN ACTION...%s', action)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', next_state_evaluation)

        return action, edge_visited_rates, best_average_evaluation, next_state_evaluation

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
        if deterministic:
            best_move_idx = np.argmax(move_probabilities)
        else:
            best_move_idx = np.random.choice(len(move_probabilities), p=move_probabilities)

        best_move = allowed_actions[best_move_idx]

        return best_move

    def get_preds(self, state, temperature=1):
        # predict the leaf
        board = state.board
        partner_board = state.partner_board

        x1 = input_representation.board_to_planes(board)
        x1 = np.expand_dims(x1, axis=0)
        x2 = input_representation.board_to_planes(partner_board)
        x2 = np.expand_dims(x2, axis=0)

        inputs = {"input_1": x1, "input_2": x2}

        # Set graph and load varialbes so model predict will work
        with self.graph.as_default():
            import tensorflow as tf
            from tensorflow.compat.v1.keras.backend import get_session
            get_session().run(tf.compat.v1.global_variables_initializer())
            get_session().run(tf.compat.v1.local_variables_initializer())

            predictions = self.model.predict(inputs)

        # value head should be one value to say how good my state is
        value_head = predictions[0]
        # policy head gives a 2272 big vector with prob for each state
        policy_head = predictions[1][0]

        allowed_action_idxs = [output_representation.move_to_policy_idx
                               (move, is_white_to_move=board.turn) for move in state.allowedActions]

        mask = np.ones(policy_head.shape, dtype=bool)
        mask[allowed_action_idxs] = False
        policy_head[mask] = -100

        # TODO problem for too high policy head...
        # TODO add temperature
        odds = np.exp(policy_head / temperature)
        move_probabilities = odds / np.sum(odds)

        allowed_actions = [output_representation.policy_idx_to_move
                           (idx, is_white_to_move=board.turn, board_id=board.board_id) for idx in allowed_action_idxs]

        # Enable to print action prob
        # self.print_action_prob(move_probabilities, allowed_actions, allowed_action_idxs)
        return value_head, move_probabilities, allowed_action_idxs, allowed_actions

    def print_action_prob(self, move_probabilities, allowed_actions, allowed_action_idxs):
        print("move probabilites: ", move_probabilities)
        print("allowed actions: ", allowed_actions)
        print("allowed actions idx: ", allowed_action_idxs)
        print("--------------")
        for i, idx in enumerate(allowed_action_idxs):
            print("Move: ", allowed_actions[i])
            print("prob: ", move_probabilities[idx])
            print("-----------")

    ####
    #
    def expand_and_evaluate_leaf(self, leaf, result, done):

        lg.logger_mcts.info('------EVALUATING LEAF------')
        if done == 0:  # game is still in process

            value_head, move_probabilities, allowed_action_idxs, allowed_actions = self.get_preds(leaf.state)
            lg.logger_mcts.info('PREDICTED VALUE_HEAD FOR %d: %f', leaf.state.playerTurn, value_head)

            # limit to allowed actions
            move_probabilities = move_probabilities[allowed_action_idxs]

            # expand tree (for all allowed actions)
            for idx, action in enumerate(allowed_actions):
                new_state, _, _ = leaf.state.take_action(action)
                if new_state.id not in self.mcts.tree:
                    node = mc.Node(new_state)
                    self.mcts.add_node(node)
                    lg.logger_mcts.info('added node...%s...p = %f', node.id, move_probabilities[idx])
                else:
                    node = self.mcts.tree[new_state.id]
                    lg.logger_mcts.info('existing node...%s...', node.id)

                new_edge = mc.Edge(leaf, node, move_probabilities[idx], action)
                leaf.edges.append((action, new_edge))

        else:  # after game is done (done ==1). in this case, do not use Neural Network,
            # but use the result of the game directly as leaf evaluation.
            value_head = result
            lg.logger_mcts.info('GAME RESULT FOR %d: %f', leaf.playerTurn, result)

        return value_head  # here was "result" before, this was probably a mistake..

    def get_statistics_of_root_edges(self):

        edges = self.mcts.root.edges
        edge_visited_rates = {}
        node_average_evaluations = {}
        rates_total = 0

        for action, edge in edges:
            # Todo will only take first argmax, but several ones in actions
            edge_visited_rate = edge.stats['node_visits']
            rates_total += edge_visited_rate
            edge_visited_rates[action] = edge_visited_rate
            node_average_evaluations[action] = edge.stats['node_average_evaluation']

        # prevent division by zero error. In case there are no edges visited the actions/edges can be chosen arbitrarily.
        if rates_total == 0:
            rates_total = 1

        for key, value in edge_visited_rates.items():
            # normalize edge_visited_rate to sum up to 1 (probability distribution)
            edge_visited_rates[key] = value / (rates_total * 1.0)

        return edge_visited_rates, node_average_evaluations

    ####
    # choose_action: pick the action where the visited rate is max. In the first few rounds:
    # choose an action with higher probability, where the visited rate is higher.
    # param: edges_visited_rates, values (map actions to their values), higher_noise (in the first few rounds the noise is higher)
    # return: action and its corresponding value
    ####

    def choose_action(self, edges_visited_rates, node_average_evaluations, higher_noise):
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

        node_average_evaluation = node_average_evaluations[action]

        # return action
        return action, node_average_evaluation

    # TODO reimplement replay and predict

    def replay(self, ltmemory):
        lg.logger_mcts.info('******RETRAINING MODEL******')

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch]), 'policy_head': np.array([row['AV'] for row in minibatch])}

            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)
            lg.logger_mcts.info('NEW LOSS %s', fit.history)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

        """
        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')

        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(1.0)
        """
        print('\n')
        # self.model.printWeightAverages()

    def build_mcts(self, state):

        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):

        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
