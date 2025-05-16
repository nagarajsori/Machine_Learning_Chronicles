class CheckersAI:
    def __init__(self, evaluator: CheckersEvaluator, max_depth=4):
        self.evaluator = evaluator
        self.max_depth = max_depth

    def choose_move(self, board: CheckersBoard, is_white_turn: bool):
        best_score = float('-inf') if is_white_turn else float('inf')
        best_move = None

        for i in range(32):
            if (is_white_turn and board.board[i] in [Piece.WHITE, Piece.WHITE_KING]) or \
               (not is_white_turn and board.board[i] in [Piece.BLACK, Piece.BLACK_KING]):
                moves = board.get_moves(i)
                for move in moves:
                    new_board = board.clone()
                    self.make_move(new_board, move)
                    score = self.minimax(new_board, self.max_depth - 1, not is_white_turn,
                                         alpha=float('-inf'), beta=float('inf'))

                    if is_white_turn and score > best_score:
                        best_score = score
                        best_move = move
                    elif not is_white_turn and score < best_score:
                        best_score = score
                        best_move = move

        return best_move

    def minimax(self, board, depth, is_white_turn, alpha, beta):
        if depth == 0:
            return self.evaluator.evaluate(board, is_white_turn)

        moves = []
        for i in range(32):
            if (is_white_turn and board.board[i] in [Piece.WHITE, Piece.WHITE_KING]) or \
               (not is_white_turn and board.board[i] in [Piece.BLACK, Piece.BLACK_KING]):
                moves += [(i, m) for m in board.get_moves(i)]

        if not moves:
            return self.evaluator.evaluate(board, is_white_turn)

        if is_white_turn:
            max_eval = float('-inf')
            for (i, m) in moves:
                new_board = board.clone()
                self.make_move(new_board, (i, m))
                eval = self.minimax(new_board, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for (i, m) in moves:
                new_board = board.clone()
                self.make_move(new_board, (i, m))
                eval = self.minimax(new_board, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def make_move(self, board: CheckersBoard, move: tuple):
        src, dst = move
        piece = board.board[src]
        board.board[src] = Piece.EMPTY
        board.board[dst] = piece

        # Handle promotion
        row, _ = board.index_to_coord(dst)
        if piece == Piece.WHITE and row == 7:
            board.board[dst] = Piece.WHITE_KING
        elif piece == Piece.BLACK and row == 0:
            board.board[dst] = Piece.BLACK_KING


# board = CheckersBoard()
# evaluator = CheckersEvaluator()
# ai = CheckersAI(evaluator, max_depth=3)

# move = ai.choose_move(board, is_white_turn=True)
# print("Best move for White:", move)
