"""
Contains class Simple_Agent
Objects of Simple_Agent class are simple Bughouse players that do not use a
trained model. The evaluation is done by a hard-coded evaluation function as defined in eval.py
"""

import numpy as np
import random

import mcts as mc
import eval
# import logger as lg



class Simple_Agent():
    ##########
    # param:
    # name - agent name
    # state size
    # action size
    # number of mcts simulations
    # cpuct - exploration coefficient for uct
    # model - the neural net. Not used in simple agent, but kept here for the purpose of later extension
    ##########
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        #self.model = model #use later

        # mcts saves tree info and statistics.
        self.mcts = None

    ##########
    # do one simulation of a game and evaluate the outcome.
    ##########
    def simulate(self):


        # lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        # self.mcts.root.state.render(lg.logger_mcts)
        # lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()
        # leaf.state.render(lg.logger_mcts)

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.back_fill(leaf, value, breadcrumbs)

    def act(self, state, tau):


        if self.mcts == None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state)#so that it doesnt have to be searched again.

        #### run the simulation
        for sim in range(self.MCTSsimulations): #TODO use fixed time instead of fixed nr of simulations
            # lg.logger_mcts.info('***************************')
            # lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            # lg.logger_mcts.info('***************************')
            self.simulate() #updates MCTS

        #### get action values
        pi, values = self.get_action_value(1)

        ####pick the action
        action, value = self.choose_action(pi, values, tau)

        nextState, _, _ = state.take_action(action)

        #NN_value = -self.get_preds(nextState)[0]
        nn_value = 0# The Neural Net is not used yet anyway. Therefore the value is set to neutral.

        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', nn_value)

        return (action, pi, value, nn_value)

    def evaluate_leaf(self, leaf, value, done, breadcrumbs):

        lg.logger_mcts.info('------EVALUATING LEAF------')
        if done == 0:

            #value, probs, allowedActions = self.get_preds(leaf.state)
            allowedActions = np.array(leaf.state.allowedActions)
            print(allowedActions)
            # In first run leaf is empty so set value to fixed value
            if leaf.edges:
                parent_edge = leaf.edges[0]# TODO: is the first edge of a node really the parent edge? Easier Alternative: use absolute evaluation fctn for the value.
                value = eval.eval_move(parent_edge.action, parent_edge.inNode)
            else:
                value = 100     # TODO what value?
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            print(allowedActions.shape)
            probs = np.ones(allowedActions.shape[0])#TODO: is this the right data type? array?
            #probs = probs[allowedActions]

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

        else:
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

        return (value, breadcrumbs)

    def get_action_value(self, tau):
        print("Tau: ", tau)

        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            # Todo will only take first argmax, but several ones in actions
            action = np.argmax(action)
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def choose_action(self, pi, values, tau):

        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def build_mcts(self, state):

        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):

        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
