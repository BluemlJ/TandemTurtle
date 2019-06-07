import numpy as np
import random
from time import sleep
import chess

# import logger as lg

from game import Game, GameState

"""
def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0,
                               goes_first=0):
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                  config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                  config.HIDDEN_CNN_LAYERS)

        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None,
                                                    goes_first)

    return (scores, memory, points, sp_scores)
"""


def play_websocket_game(player, logger, interface, turns_until_tau0, goes_first):
    while interface.color is None:
        sleep(0.01)
    env = Game(0)
    state = env.reset()
    turn = 0
    done = False

    while not done:
        # wait till game started
        while not interface.isMyTurn:
            sleep(0.01)

        # perform move of other player
        if (turn > 0 and interface.color == 'white') or interface.color == 'black':
            interface.logViaInterfaceType(f"[{player.name}] performing action of opponent {interface.lastMove}")
            mv = chess.Move.from_uci(interface.lastMove)
            mv.board_id = 0
            state, value, done, _ = env.step(mv)
            interface.lastMove = ''
            for move in interface.otherMoves:
                mv = chess.Move.from_uci(move)
                mv.board_id = 1
                state.push_action(mv)
            interface.otherMoves = []

        interface.logViaInterfaceType(f"[{player.name}] It's my turn!")

        turn += 1
        tauNotReached = 1 if turn < turns_until_tau0 else 0

        # get action
        action, pi, MCTS_value, NN_value = player.act(state, tauNotReached)

        # send message
        logger.info(f"move {action} was played by {player.name}")
        interface.sendAction(action)
        interface.isMyTurn = False

        # Do the action
        state, value, done, _ = env.step(action)

        # the value of the newState from the POV of the new playerTurn
        # i.e. -1 if the previous player played a winning move

        env.gameState.render(logger)

    print(f"[{player.name}] Game finished!")