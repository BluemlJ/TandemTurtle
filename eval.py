"""
Evaluation function to return the value of one given leaf. One option is to start by using pre defined evaluation tables,
which later can be augmented to improve performance.

Idea: Build function with following properties:

"""

# Eval table and stuff copy pasted from sunfish
pieces = {'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000}
pst = {'P': (0, 0, 0, 0, 0, 0, 0, 0,
             78, 83, 86, 73, 102, 82, 85, 90,
             7, 29, 21, 44, 40, 31, 44, 7,
             -17, 16, -2, 15, 14, 0, 15, -13,
             -26, 3, 10, 9, 6, 1, 0, -23,
             -22, 9, 5, -11, -10, -2, 3, -19,
             -31, 8, -7, -37, -36, -14, 3, -31,
             0, 0, 0, 0, 0, 0, 0, 0),
       'N': (-66, -53, -75, -75, -10, -55, -58, -70,
             -3, -6, 100, -36, 4, 62, -4, -14,
             10, 67, 1, 74, 73, 27, 62, -2,
             24, 24, 45, 37, 33, 41, 25, 17,
             -1, 5, 31, 21, 22, 35, 2, 0,
             -18, 10, 13, 22, 18, 15, 11, -14,
             -23, -15, 2, 0, 2, 0, -23, -20,
             -74, -23, -26, -24, -19, -35, -22, -69),
       'B': (-59, -78, -82, -76, -23, -107, -37, -50,
             -11, 20, 35, -42, -39, 31, 2, -22,
             -9, 39, -32, 41, 52, -10, 28, -14,
             25, 17, 20, 34, 26, 25, 15, 10,
             13, 10, 17, 23, 17, 16, 0, 7,
             14, 25, 24, 15, 8, 25, 20, 15,
             19, 20, 11, 6, 7, 6, 20, 16,
             -7, 2, -15, -12, -14, -15, -10, -10),
       'R': (35, 29, 33, 4, 37, 33, 56, 50,
             55, 29, 56, 67, 55, 62, 34, 60,
             19, 35, 28, 33, 45, 27, 25, 15,
             0, 5, 16, 13, 18, -4, -9, -6,
             -28, -35, -16, -21, -13, -29, -46, -30,
             -42, -28, -42, -25, -25, -35, -26, -46,
             -53, -38, -31, -26, -29, -43, -44, -53,
             -30, -24, -18, 5, -2, -18, -31, -32),
       'Q': (6, 1, -8, -104, 69, 24, 88, 26,
             14, 32, 60, -10, 20, 76, 57, 24,
             -2, 43, 32, 60, 72, 63, 43, 2,
             1, -16, 22, 17, 25, 20, -13, -6,
             -14, -15, -2, -5, -1, -10, -20, -22,
             -30, -6, -13, -11, -16, -11, -16, -27,
             -36, -18, 0, -19, -15, -15, -21, -38,
             -39, -30, -31, -13, -31, -36, -34, -42),
       'K': (4, 54, 47, -99, -99, 60, 83, -62,
             -32, 10, 55, 56, 56, 55, 10, 3,
             -62, 12, -57, 44, -67, 28, 37, -31,
             -55, 50, 11, -4, -19, 13, 0, -49,
             -55, -43, -52, -28, -51, -47, -8, -50,
             -47, -42, -43, -79, -64, -32, -29, -32,
             -4, 3, -14, -50, -57, -18, 13, 4,
             17, 30, -3, -14, 6, -1, 40, 18),
       }

# adds value of each pice to correspondang table on every value
for k, table in pst.items():
    def padrow(row): return (0,) + tuple(x + pieces[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8:i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20


def simple_eval_gamestate(state):
    nr_pieces_current_player = sum(len(list(state.board.pieces(i, state.player_turn))) for i in range(1, 7))
    nr_pieces_opponent = sum(len(list(state.board.pieces(i, state.player_turn * (-1)))) for i in range(1, 7))
    score = nr_pieces_current_player / nr_pieces_opponent
    return score


def eval_move(move, board):
    """
    Evaluation function
    Calc difference of value before move and after move

    :param move: move to eval in format: prev pos, new pos TODO add
    :param board: board in format XXX, where to eval move on  TODO add
    :return: Value of move
    """
    print(move)
    print(board)
    print(board.state)
    print(board.state.board)

    print("NOT USEFULL ATM")
    i, j = move
    piece, q = board[i], board[j]
    # Score difference of piece moving
    score = pst[piece][j] - pst[piece][i]
    # Capture
    if q.is_enemy():
        # enemy_piece_value = flip board and eval piece val at spot
        score += enemy_picec_value
    # Castling check detection
    if move_is_castle:
        score += score_for_castling
    # Castling
    if piece == 'K' and abs(i - j) == 2:
        score += pst['R'][(i + j) // 2]
        score -= pst['R'][A1 if j < i else H1]
    # Special pawn stuff
    if piece == 'P':
        score += value_afer_transforming
        if A8 <= j <= H8:
            score += pst['Q'][j] - pst['P'][j]
        if j == self.ep:
            score += pst['P'][119 - (j + S)]
    return score
