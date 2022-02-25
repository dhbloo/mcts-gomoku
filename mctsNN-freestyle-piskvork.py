import time


def visit(model, data):
    value, policy = model(data)
    winrate = (value[0] - value[1] + 1) / 2
    return winrate, policy


def main(model_file, load_type, device, **kwsearchargs):
    # load modules after recording start time
    start_time = time.time()
    import pisqpipe as pp
    import neuralnet as nn
    from search import Board, MCTS

    model = nn.load_model(load_type, model_file, device)
    board = None

    def init():
        if pp.width != 15 or pp.height != 15:
            pp.pipeOut("ERROR size of the board")
            return
        nonlocal board
        board = Board(pp.width, pp.height)
        pp.pipeOut("OK")

    def restart():
        nonlocal board
        board = Board(board.width, board.height)
        pp.pipeOut("OK")

    def putmove(x, y):
        nonlocal board
        board.move(x, y)

    def takeback(x, y):
        nonlocal board
        board.undo()

    def turn():
        if pp.terminateAI:
            return

        nonlocal board, start_time
        assert board is not None
        mcts = MCTS(lambda data: visit(model, data), **kwsearchargs)
        turn_time = min(pp.info_timeout_turn / 1000, pp.info_time_left / 7 / 1000) - 0.03

        if start_time is not None:
            tic = start_time
            start_time = None
        else:
            tic = time.time()
        visits = 0
        while not visits or time.time() - tic < turn_time:
            mcts.search(board)
            visits += 1
            if pp.terminateAI:
                break
        child, winrate, num_visits = mcts.choose(board)
        move = child.last_move
        toc = time.time()

        pp.pipeOut(f"MESSAGE winrate: {winrate:.4f}, " +
                   f"bestmove: {chr(move[0] + ord('A'))}{move[1] + 1} ({num_visits} v), " +
                   f"visits: {visits}, " + f"time: {toc-tic:.4f}s, " +
                   f"v/s: {visits/(toc-tic):.1f}")
        pp.do_mymove(*move)

    def end():
        pass

    pp.brain_init = init
    pp.brain_restart = restart
    pp.brain_my = putmove
    pp.brain_opponents = putmove
    pp.brain_takeback = takeback
    pp.brain_turn = turn
    pp.brain_end = end
    pp.main()


if __name__ == "__main__":
    args = {
        'model_file': './data/export_onnx_resnet_basic-nostm_20b256fv0_01000000.onnx',
        'load_type': 'onnx',
        'device': 'cpu',
    }
    main(**args)
