import config
import util.nn_interface as nni
import game_play
from agent import Agent
from game.game import Game


print("start testig")
print("loading neural network")
model = nni.load_nn()
print("creating Game")
env = Game(0)
print("creating Players")

player1 = Agent("player1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, None)
player2 = Agent("player2", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, None)

print("play Matches")
game_play.playMatches(player1, player2, config.EPISODES, config.TURNS_WITH_HIGH_NOISE)
