from enum import Enum
import numpy as np

class Piece(Enum):
    EMPTY = 0
    WHITE = 1
    WHITE_KING = 2
    BLACK = -1
    BLACK_KING = -2

class CheckersBoard:
    def __init__(self):
        # Only 32 usable positions (dark squares)
        self.board = [Piece.EMPTY] * 32
        self.initialize_board()

    def initialize_board(self):
        for i in range(12):
            self.board[i] = Piece.BLACK
        for i in range(12, 20):
            self.board[i] = Piece.EMPTY
        for i in range(20, 32):
            self.board[i] = Piece.WHITE

    def print_board(self):
        def piece_str(p): return {
            Piece.EMPTY: '.',
            Piece.WHITE: 'w',
            Piece.WHITE_KING: 'W',
            Piece.BLACK: 'b',
            Piece.BLACK_KING: 'B'
        }[p]

        rows = []
        for i in range(0, 32, 4):
            row = " ".join(piece_str(self.board[i + j]) for j in range(4))
            rows.append(row)
        for r in reversed(rows):
            print(r)

# Test
board = CheckersBoard()
board.print_board()
