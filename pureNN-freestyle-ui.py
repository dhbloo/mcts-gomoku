import time
import neuralnet as nn
from search import Board


def next_move(board, model, data, device):
    value, policy = nn.eval_model(model, data, device)
    policy = policy.flatten()

    # calc winrate, drawrate and best valid move
    winrate = (value[0] - value[1] + 1) / 2
    drawrate = value[2]

    movelist = policy.argsort(descending=True)
    for move in movelist:
        move_x, move_y = move % board.width, move // board.width
        if board.is_legal(move_x, move_y):
            bestmove = move
            break
    bestmove_x, bestmove_y = bestmove % board.width, bestmove // board.width

    return winrate, drawrate, (bestmove_x, bestmove_y)


def input_move():
    input_str = input("Input your move (empty for AI move): ")
    if input_str == "":
        return None
    x, y = input_str[0].upper(), input_str[1:]
    return ord(x) - ord('A'), int(y) - 1


def test_play(model_file, load_type, device, board_width, board_height):
    model = nn.load_model(load_type, model_file, device)
    board = Board(board_width, board_height)

    while not board.is_terminal():
        print(board)
        move = input_move()
        if move is None:
            tic = time.time()
            winrate, drawrate, move = next_move(board, model, board.get_data(), device)
            toc = time.time()
            message = (f"winrate: {winrate:.4f}, drawrate: {drawrate:.4f}, " +
                       f"bestmove: {chr(move[0] + ord('A'))}{move[1] + 1}, " +
                       f"time: {(toc-tic)*1000:.2f}ms")
            print(message)
        board.move(*move)
    print(board)


if __name__ == "__main__":
    args = {
        'model_file': './data/export_jit_resnet_basic-nostm_15b192fv0_00500000.pth',
        'load_type': 'jit',
        'device': 'cpu',
        'board_width': 15,
        'board_height': 15,
    }
    test_play(**args)
