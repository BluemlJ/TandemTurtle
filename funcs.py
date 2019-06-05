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
    while interface.color == None:
        sleep(0.01)
    env = Game(0)
    env2 = Game(1)
    state = env.reset()
    state2 = env2.reset()
    turn = 0
    done = False

    while not done:
        # wait till game started
        while not interface.isMyTurn:
            sleep(0.01)

        # perform move of other player
        if (turn > 0 and interface.color == 'white') or interface.color == 'black':
            print(f"[{player.name}] performing action of opponent {interface.lastMove}")
            mv = chess.Move.from_uci(interface.lastMove)
            mv.board_id = 0
            state, value, done, _ = env.step(mv)
            interface.lastMove = ''
            for move in interface.otherMoves:
                mv = chess.Move.from_uci(move)
                mv.board_id = 1
                state2, value2, done2, _ = env2.step(mv)
            interface.otherMoves = []

        
        print(f"[{player.name}] It's my turn!")
        
        turn += 1
        tauNotReached = 1 if turn < turns_until_tau0 else 0

        # get action
        action, pi, MCTS_value, NN_value = player.act(state, tauNotReached)

        # send message
        logger.info(f"move {action} was played by {player.name}")
        print(action)
        interface.sendAction(action)
        interface.isMyTurn = False

        # Do the action
        state, value, done, _ = env.step(action)

        # the value of the newState from the POV of the new playerTurn
        # i.e. -1 if the previous player played a winning move

        env.gameState.render(logger)
    
    print(f"[{player.name}] Game finished!")



def play_matches(players, number_games, logger, turns_until_tau0, player1_goes_first=None):
    # TODO include player3, 4 and do something with env2
    player1 = players[0]
    player2 = players[2]
    player3 = players[1]
    player4 = players[3]

    env1 = Game(0)
    env2 = Game(1)
    scores = {player1.name: 0, "drawn": 0, player2.name: 0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}

    for e in range(number_games):

        # logger.info('====================')
        # logger.info('EPISODE %d OF %d', e + 1, number_games)
        # logger.info('====================')

        print(str(e + 1) + ' ', end='')

        state = env1.reset()

        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if player1_goes_first is None:
            player1Starts = bool(random.getrandbits(1))
        else:
            player1Starts = player1_goes_first

        # Get color of players and determine the starter

        if player1Starts:
            players = {1: {"agent": player1, "name": player1.name}, -1: {"agent": player2, "name": player2.name}}
            logger.info(player1.name + ' plays as X')
        else:
            players = {1: {"agent": player2, "name": player2.name}, -1: {"agent": player1, "name": player1.name}}
            logger.info(player2.name + ' plays as X')
            logger.info('--------------')

        env1.gameState.render(logger)

        while done == 0:
            turn = turn + 1

            # Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)

            # send message
            logger.info(f"move {action} was played by {players[state.playerTurn]['name']}")
            players[state.playerTurn]['agent'].interface.sendAction(action)

            # Do the action
            state, value, done, _ = env1.step(action)

            # the value of the newState from the POV of the new playerTurn
            # i.e. -1 if the previous player played a winning move

            env1.gameState.render(logger)

            # Updates scores and loggs if someone won
            if done == 1:
                if value == 1:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1

                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])
    return scores, points, sp_scores
