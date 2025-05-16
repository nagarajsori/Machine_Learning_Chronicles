class CheckersBoard:
    def __init__(self):
        self.board = [Piece.EMPTY] * 32
        self.initialize_board()

    def initialize_board(self):
        for i in range(12):
            self.board[i] = Piece.BLACK
        for i in range(20, 32):
            self.board[i] = Piece.WHITE

    def clone(self):
        new_board = CheckersBoard()
        new_board.board = self.board[:]
        return new_board

    def is_king(self, piece):
        return piece in [Piece.WHITE_KING, Piece.BLACK_KING]

    def is_enemy(self, piece, other):
        if piece == Piece.EMPTY or other == Piece.EMPTY:
            return False
        return (piece.value * other.value) < 0

    def get_moves(self, index):
        """
        Returns a list of legal move targets for the piece at board[index]
        """
        piece = self.board[index]
        if piece == Piece.EMPTY:
            return []

        directions = []
        if piece in [Piece.WHITE, Piece.WHITE_KING]:
            directions += [(-4, -3), (-5, -4)]  # Forward-left, forward-right
        if piece in [Piece.BLACK, Piece.BLACK_KING]:
            directions += [(4, 5), (3, 4)]  # Backward-left, backward-right

        if self.is_king(piece):
            directions += [(-4, -3), (-5, -4), (4, 5), (3, 4)]

        legal_moves = []
        for offset, jump in directions:
            target = index + offset
            if 0 <= target < 32 and self.board[target] == Piece.EMPTY:
                legal_moves.append((index, target))
            elif 0 <= target < 32 and self.is_enemy(piece, self.board[target]):
                landing = index + jump
                if 0 <= landing < 32 and self.board[landing] == Piece.EMPTY:
                    legal_moves.append((index, landing))  # jump

        return legal_moves

    def print_board(self):
        def piece_str(p): return {
            Piece.EMPTY: '.',
            Piece.WHITE: 'w',
            Piece.WHITE_KING: 'W',
            Piece.BLACK: 'b',
            Piece.BLACK_KING: 'B'
        }[p]

        for row in range(7, -1, -1):
            row_str = ''
            for col in range(8):
                idx = self.coord_to_index(row, col)
                if idx is not None:
                    row_str += piece_str(self.board[idx]) + ' '
                else:
                    row_str += '  '
            print(row_str)

    def coord_to_index(self, row, col):
        if (row + col) % 2 == 0:
            return None
        return (row * 4) + (col // 2)

    def index_to_coord(self, idx):
        row = idx // 4
        col = (idx % 4) * 2 + ((row + 1) % 2)
        return row, col
