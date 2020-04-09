from othello.model import Board, ScoreBoard


def judge(board: Board) -> int:
    return board.board.sum()


def simple_score(board: Board, score: ScoreBoard) -> float:
    return (board.board * score.board).sum()
