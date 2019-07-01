import threading
from time import sleep

import config
import util.nn_interface as nni
from agent import Agent
from game.game import Game
from pretraining.nn import NeuralNetwork


class Shared_state(object):
    def __init__(self):
        self.moves_A = []
        self.moves_B = []
        self.moves = [self.moves_A, self.moves_B]
        self.color_A = 1
        self.color_B = 1
        self.colors = [self.color_A, self.color_B]
        self.turn_A = 0
        self.turn_B = 0
        self.turns = [self.turn_A, self.turn_B]
        self.done = False

    def reset(self):
        self.moves_A = []
        self.moves_B = []
        self.color_A = 1
        self.color_B = 1
        self.turn_A = 0
        self.turn_B = 0


class Local_agent_thread(threading.Thread):
    def __init__(self, agent, turns_until_tau0, shared_state, board_id, is_white, *args, **kwargs):
        super(Local_agent_thread, self).__init__(*args, **kwargs)
        self.shared_state = shared_state
        self.turns_until_tau0 = turns_until_tau0
        self.agent = agent
        self.board_id = board_id
        self.partner_board_id = 0 if board_id == 1 else 1
        self.env = Game(board_id)
        self.state = self.env.reset()
        self.color = 1 if is_white else -1

    def run(self):
        last_partner_move = 0

        while True:
            print(id(self.shared_state))
            # agent waits for its turn
            while not self.shared_state.colors[self.board_id] == self.color:
                sleep(0.01)

            # push moves on the partner board
            while last_partner_move < len(self.shared_state.moves[self.partner_board_id]):
                self.state.push_action(self.shared_state.moves[self.partner_board_id][last_partner_move],
                                       self.partner_board_id)
                last_partner_move += 1

            # push opponents move
            if len(self.shared_state.moves[self.board_id]) > 0:
                self.state.push_action(self.shared_state.moves[self.board_id][-1])

            if not self.shared_state.done:
                self.shared_state.turns[self.board_id] += 1
                if self.shared_state.turns[self.board_id] < self.turns_until_tau0:
                    action, pi, MCTS_value, NN_value = self.agent.act(self.state, 1)
                else:
                    action, pi, MCTS_value, NN_value = self.agent.act(self.state, 0)

                self.state.push_action(action)
                if self.state.isEndGame:
                    self.shared_state.done = True
                print(self.state.boards)
                self.shared_state.moves[self.board_id].append(action)
                self.shared_state.colors[self.board_id] = self.shared_state.colors[self.board_id] * -1
            else:
                break


env = Game(0)
shared_state = Shared_state()
model = NeuralNetwork().model
model._make_predict_function()
turns_until_tau0 = config.TURNS_WITH_HIGH_NOISE

agent1 = Agent("Agent 1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, None)
agent2 = Agent("Agent 1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, None)
agent3 = Agent("Agent 1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, None)
agent4 = Agent("Agent 1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, None)
threads = [
    Local_agent_thread(agent1, turns_until_tau0, shared_state, 0, True),
    Local_agent_thread(agent1, turns_until_tau0, shared_state, 0, False),
    Local_agent_thread(agent1, turns_until_tau0, shared_state, 1, True),
    Local_agent_thread(agent1, turns_until_tau0, shared_state, 1, False)
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
