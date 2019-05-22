import numpy as np
np.set_printoptions(suppress=True)




from game import Game, GameState
import funcs

import logger as lg

import config
from simple_agent import Simple_Agent


# TODO created twice why not give as parameter
env = Game(0)


agent1 = Simple_Agent('agent1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None)
agent2 = Simple_Agent('agent2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None)
agent3 = Simple_Agent('agent3', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None)
agent4 = Simple_Agent('agent4', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None)
agents = [agent1, agent2, agent3, agent4]

funcs.play_matches(agents, config.EPISODES, lg.logger_main,
                   turns_until_tau0=config.TURNS_UNTIL_TAU0)

exit()
