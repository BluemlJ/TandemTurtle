from time import sleep
import chess
import util.logger as lg
from game.game import Game, GameState
import random


def playMatches(player_agents, EPISODES, turns_until_high_noise, memory=None, goes_first=0):
    # TODO: test and reimplement
    env = Game(0)
    scores = {}  # TODO Scores for each player
    points = {"best_players": [], "new_players": []}

    for e in range(EPISODES):
        print('====================')
        print('EPISODE %d OF %d', e + 1, EPISODES)
        print('====================')
        print(str(e + 1) + ' ', end='')

        state = env.reset()

        done = 0
        turn = 0
        for agent in player_agents:
            agent.mcts = None

        players = {1: {"agent": player_agents[0], "name": "best1"},
                   2: {"agent": player_agents[1], "name": "best2"},
                   3: {"agent": player_agents[2], "name": "new1"},
                   4: {"agent": player_agents[3], "name": "new2"}
                   }

        # TODO who goes first

        while not done:
            turn = turn + 1

            # TODO run MCTS and return an action

            if memory is not None:
                print('action: %d', action)
                print('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)],
                            MCTS_value, 2)
                print('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)], NN_value, 2)
                print('====================')

                # Commit the move to memory
                # smthing like this -> memory.commit_stmemory(env.name, state, pi)

            # TODO Do the action
            state, value, done, _ = env.step(
                action)  # the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move

            if done == 1:
                if memory is not None:
                    # If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value

                    memory.commit_ltmemory()

                # TODO add 1 to the winning team and add 1 for a draw

    return (scores, memory, points)


def play_websocket_game(player, logger, interface, turns_with_high_noise, goes_first):
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
        higher_noise = 1 if turn < turns_with_high_noise else 0

        # get action, edge_visited_rates, best_average_evaluation, next_state_evaluation
        action, _, _, _ = player.act(state, higher_noise)

        # send message
        lg.logger_model.info(f"move {action} was played by {player.name}")
        interface.sendAction(action)
        interface.isMyTurn = False

        # Do the action
        state, value, done, _ = env.step(action)

        # the value of the newState from the POV of the new playerTurn
        # i.e. -1 if the previous player played a winning move

    print(f"[{player.name}] Game finished!")
