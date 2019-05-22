"""
Contains class Simple_Agent
Objects of Simple_Agent class are simple Bughouse players that do not use a
trained model. The evaluation is done by a hard-coded evaluation function as defined in eval.py
"""

import numpy as np
import random

import mcts as mc
import eval
import logger as lg



class Simple_Agent():
    ##########
    # param:
    # name - agent name
    # state size
    # action size
    # number of MCTS simulations
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
    # do one simulation of a game and evaluate the outcome. Update the MC search tree in the progress.
    ##########
    def simulate(self):

        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        self.mcts.root.state.render(lg.logger_mcts) # log game state
        lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()
        leaf.state.render(lg.logger_mcts)

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.back_fill(leaf, value, breadcrumbs)

    ####
    # act - run simulations updating the MC-search-tree. Then pick an action.
    # param:
    # state - the game state
    # tau - 1 (in the beginning, when the moves are not yet deterministic)
    #       or 0 (after some time, when the simple_agent starts playing deterministically.
    # returns:
    # action - the chosen action,
    # pi - the priorities (?) of the different actions,
    # value - (?)
    # nn_value - zero in this simple case
    # .

    def act(self, state, tau):
        print("act with tau = %d",tau)
        #### go to mcts node that corresponds to state or build new mcts.
        if self.mcts == None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state)#so that it can use previous simulations

        #### run the simulation
        for sim in range(self.MCTSsimulations): #TODO (later) use fixed time instead of fixed nr of simulations
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('***************************')
            self.simulate() #updates MCTS

        #### get action values. pi is a probability distribution over the visited nodes.
        # pi, values = self.get_action_values(1)
        pi, _ = self.get_action_values(1)

        #### pick the action where pi is max.
        # action, value = self.choose_action(pi, values, tau) #Todo (later) what do we need the value for?
        action = self.choose_action(pi, None, tau)

        #nextState, _, _ = state.take_action(action) #only needed for nn_value

        #NN_value = -self.get_preds(nextState)[0]
        nn_value = 0# The Neural Net is not used yet anyway. Therefore the value is set to neutral.

        lg.logger_mcts.info('ACTION VALUES...%s', pi)
        lg.logger_mcts.info('CHOSEN ACTION...%s', action)
        # lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', nn_value)

        # return (action, pi, value, nn_value)
        return (action, pi, None, nn_value)

    ####
    # evaluate_leaf: .
    def evaluate_leaf(self, leaf, eval_value, done, breadcrumbs):#TODO (later): delete breadcrumbs, its not used, is it?

        lg.logger_mcts.info('------EVALUATING LEAF------')
        if done == 0:

            allowedActions = np.array(leaf.state.allowedActions) # no rollouts. use static eval fct instead
            print(allowedActions)

            # In first run leaf is empty so set value to fixed value
            if leaf.edges:
                parent_edge = leaf.edges[0]# TODO: is the first edge of a node really the parent edge? Easier Alternative: use absolute evaluation fctn for the value.
                eval_value = eval.eval_move(parent_edge.action, parent_edge.inNode)
                eval.simple_eval_gamestate(leaf.state)
            else:
                eval_value = 100     # TODO (later) what value?
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, eval_value)

            # no rollouts. use static eval fctn

            print("allowedActions.shape = ", allowedActions.shape)
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

        else:#after evaluation is done (done ==1)
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, eval_value)

        return (eval_value, breadcrumbs)

    def get_action_values(self, tau): #TODO: get rid of this tau or give it a better name after we know what it does
        print("Tau: ", tau)

        edges = self.mcts.root.edges
        # old:
        # pi = np.zeros(self.action_size, dtype=np.integer)
        # values = np.zeros(self.action_size, dtype=np.float32)
        pi = {}
        values = {}
        pi_total = 0

        for action, edge in edges:
            # Todo will only take first argmax, but several ones in actions
            pi_val = pow(edge.stats['N'], 1/tau) #TODO (later) why not use p[action = edge.stats['N'] directly?
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
    # param: pi, values (map actions to their values), tau
    # return: action and its corresponding value
    ####
    def choose_action(self, pi, values, tau):
        inverse = [(value, key) for key, value in pi.items()]

        if tau == 0:    # deterministic
            actions = [x[1] for i, x in enumerate(inverse) if x[0] == max(inverse)[0]]
            action = random.choice(actions)     # break ties randomly.
        else:
            pi_values = [value for value, key in inverse]
            value_idx_arr = np.random.multinomial(1, pi_values)
            value_idx = np.where(value_idx_arr == 1)[0][0]
            action = inverse[value_idx][1]

        # value = values[action]

        return action
        # return action, value

    def build_mcts(self, state):

        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):

        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
