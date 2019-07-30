import _thread
import time
from time import sleep
import chess
import util.logger as lg
from game.game import Game, GameState
import random
import subprocess
import config

from util.xboardInterface import XBoardInterface
from main import create_and_run_agent


def playMatches(best_model, new_model, EPISODES, turns_until_high_noise, memory=None):
    # TODO: test and reimplement
    env = Game(0)
    scores = {}  # TODO Scores for each player

    for e in range(EPISODES):
        print('====================')
        print('EPISODE %d OF %d', e + 1, EPISODES)
        print('====================')
        print(str(e + 1) + ' ', end='')

        server = subprocess.Popen(["node", "index.js"], cwd="../tinyChessServer", stdout=subprocess.PIPE)
        sleep(1)

        _thread.start_new_thread(create_and_run_agent, ("Agent 1", True, env, best_model, server))
        _thread.start_new_thread(create_and_run_agent, ("Agent 2", False, env, new_model))
        _thread.start_new_thread(create_and_run_agent, ("Agent 3", True, env, new_model))
        _thread.start_new_thread(create_and_run_agent, ("Agent 4", False, env, best_model))

        # TODO count the score and points
    return (scores, memory)


def play_websocket_game(player, logger, interface, turns_with_high_noise, is_random=False):
    while interface.color is None:
        sleep(0.01)

    env = Game(0)
    state = env.reset()
    player.build_mcts(state)
    turn = 0
    done = False
    used_time = 0
    while not done:
        # wait till game started
        while not interface.isMyTurn and not interface.done:
            sleep(0.01)
        if interface.done:
            break
        turn_start_time = time.time()
        # perform move of other player
        if (turn > 0 and interface.color == 'white') or interface.color == 'black':
            interface.logViaInterfaceType(f"[{player.name}] performing action of opponent {interface.lastMove}")
            mv = chess.Move.from_uci(interface.lastMove)
            mv.board_id = 0
            state, value, done, _ = env.step(mv)
            player.play_move(mv, on_partner_board=False)
            interface.lastMove = ''
            for move in interface.otherMoves:
                mv = chess.Move.from_uci(move)
                mv.board_id = 1
                state, value, done, _ = env.step(mv)
                player.play_move(mv, on_partner_board=True)
            interface.otherMoves = []
        used_time += time.time() - turn_start_time
        turn_start_time = time.time()
        if interface.done or done:
            break

        interface.logViaInterfaceType(f"[{player.name}] It's my turn!")
        turn += 1
        higher_noise = 1 if turn < turns_with_high_noise else 0
        time_left = interface.time - used_time
        print(time_left)

        # get action, edge_visited_rates, best_average_evaluation, next_state_evaluation
        if is_random:
            sleep(config.DELAY_FOR_RANDOM)
            action = player.act_random(state)
        else:
            if config.RUN_ON_NN_ONLY or (time_left < config.LOW_TIME_THRESHOLD):
                action = player.act_nn(state, higher_noise)
            else:
                action = player.suggest_move(higher_noise)
                player.play_move(action, on_partner_board=False)

        # send message
        lg.logger_model.info(f"move {action} was played by {player.name}")
        interface.sendAction(action)
        interface.isMyTurn = False
        used_time += time.time() - turn_start_time

        # Do the action
        state, value, done, _ = env.step(action)
        lg.logger_main.info(f"Turn number: {turn}")
        env.gameState.render(lg.logger_main)

        # the value of the newState from the POV of the new playerTurn
        # i.e. -1 if the previous player played a winning move

    print(f"[{player.name}] Game finished!")
    import os
    import sys
    os.execl(sys.executable, sys.executable, *sys.argv)
