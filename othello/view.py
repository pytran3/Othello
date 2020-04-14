from othello.model import Board


def view_board(board: Board) -> str:
    lines = [" abcdefgh"]
    for i in range(8):
        line = str(i + 1) + "".join([["-", "x", "o"][i] for i in board.board[i]])
        lines.append(line)
    return "\n".join(lines)
