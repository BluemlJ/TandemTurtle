from time import sleep
import chess
from game.game import Game, GameState


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
        logger.info(f"move {action} was played by {player.name}")
        interface.sendAction(action)
        interface.isMyTurn = False

        # Do the action
        state, value, done, _ = env.step(action)

        # the value of the newState from the POV of the new playerTurn
        # i.e. -1 if the previous player played a winning move

        env.gameState.render(logger)

    print(f"[{player.name}] Game finished!")