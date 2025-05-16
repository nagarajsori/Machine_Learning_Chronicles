class CheckersEvaluator:
    def __init__(self, weights=None):
        # Default weights: [material, kings, center control, mobility]
        self.weights = weights if weights else [1.0, 1.5, 0.5, 0.3]

    def evaluate(self, board: CheckersBoard, is_white_turn: bool) -> float:
        material = self.material_diff(board)
        kings = self.king_diff(board)
        center = self.center_control(board)
        mobility = self.mobility(board, is_white_turn)

        features = [material, kings, center, mobility]
        score = sum(w * f for w, f in zip(self.weights, features))
        return score if is_white_turn else -score

    def material_diff(self, board):
        white = sum(1 for p in board.board if p == Piece.WHITE)
        black = sum(1 for p in board.board if p == Piece.BLACK)
        return white - black

    def king_diff(self, board):
        white = sum(1 for p in board.board if p == Piece.WHITE_KING)
        black = sum(1 for p in board.board if p == Piece.BLACK_KING)
        return white - black

    def center_control(self, board):
        center_idxs = [13, 14, 17, 18]
        score = 0
        for i in center_idxs:
            p = board.board[i]
            if p == Piece.WHITE or p == Piece.WHITE_KING:
                score += 1
            elif p == Piece.BLACK or p == Piece.BLACK_KING:
                score -= 1
        return score

    def mobility(self, board, is_white_turn):
        player = Piece.WHITE if is_white_turn else Piece.BLACK
        total_moves = 0
        for i, p in enumerate(board.board):
            if p == player or (is_white_turn and p == Piece.WHITE_KING) or (not is_white_turn and p == Piece.BLACK_KING):
                moves = board.get_moves(i)
                total_moves += len(moves)
        return total_moves


# board = CheckersBoard()
# evaluator = CheckersEvaluator()
# score = evaluator.evaluate(board, is_white_turn=True)
# print("Board Score:", score)
