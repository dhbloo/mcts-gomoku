import time
import neuralnet as nn
from search import Board, MCTS


def visit(model, data):
    value, policy = model(data)
    winrate = (value[0] - value[1] + 1) / 2
    return winrate, policy


def input_move():
    input_str = input("Input your move (empty for AI move): ")
    if input_str == "":
        return None
    x, y = input_str[0].upper(), input_str[1:]
    return ord(x) - ord('A'), int(y) - 1


def test_play(model_file,
              load_type,
              device,
              board_width,
              board_height,
              turn_time=1,
              **kwsearchargs):
    model = nn.load_model(load_type, model_file, device)
    board = Board(board_width, board_height)
    mcts = MCTS(lambda data: visit(model, data), **kwsearchargs)

    while not board.is_terminal():
        print(board)
        move = input_move()
        if move is None:
            visits = 0
            tic = time.time()
            while not visits or time.time() - tic < turn_time:
                mcts.search(board)
                visits += 1
            child, winrate, num_visits = mcts.choose(board)
            move = child.last_move
            toc = time.time()

            message = (f"winrate: {winrate:.4f}, " +
                       f"bestmove: {chr(move[0] + ord('A'))}{move[1] + 1} ({num_visits} v), " +
                       f"visits: {visits}, " + f"time: {toc-tic:.4f}s, " +
                       f"v/s: {visits/(toc-tic):.1f}")
            print(message)
        board.move(*move)


if __name__ == "__main__":
    args = {
        'model_file': './data/export_onnx_resnet_basic-nostm_20b256fv0_01000000.onnx',
        'load_type': 'onnx',
        'device': 'cpu',
        'board_width': 15,
        'board_height': 15,
        'turn_time': 5,
    }
    test_play(**args)
