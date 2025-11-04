# gomoku_ai.py
# latest version

import numpy as np
import math
import time
from typing import List, Tuple, Optional

BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

# -------------------------
# I/O
# -------------------------
def load_board_from_txt(path: str) -> np.ndarray:
    """
    读取18x18的txt，元素以空格或不间断分隔均可。
    每行应包含18个数字（0/1/2）。
    返回 numpy int array shape (18,18)
    """
    board = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 尝试以空格分割，否则按每个字符
            parts = line.split()
            if len(parts) == BOARD_SIZE:
                row = [int(x) for x in parts]
            elif len(line) >= BOARD_SIZE:
                row = [int(ch) for ch in line[:BOARD_SIZE]]
            else:
                raise ValueError(f"行格式错误: {line}")
            if len(row) != BOARD_SIZE:
                raise ValueError("每行必须有 19 个数字")
            board.append(row)
    if len(board) != BOARD_SIZE:
        raise ValueError("文件必须有 19 行棋盘数据")
    return np.array(board, dtype=int)

def save_board_to_txt(board: np.ndarray, path: str):
    with open(path, 'w') as f:
        for r in range(board.shape[0]):
            f.write(' '.join(str(int(x)) for x in board[r]) + "\n")

def print_board(board: np.ndarray):
    print("   " + " ".join(f"{i:2d}" for i in range(board.shape[1])))
    for i, row in enumerate(board):
        print(f"{i:2d} " + " ".join({0:'.',1:'X',2:'O'}[int(x)] for x in row))

# -------------------------
# 规则与检验
# -------------------------
DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)] # 用于检验是否获胜 4个方向

# 棋子在棋盘中的位置
def in_board(x:int,y:int)->bool:
    return 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE

def check_win(board: np.ndarray, last_move: Optional[Tuple[int,int]]=None) -> int:
    """
    如果某方胜利返回棋子（1或2），否则返回0
    如果给定 last_move 可以更快判断（否则扫描全盘）
    """
    if last_move:
        sx, sy = last_move
        piece = board[sx,sy]
        if piece==EMPTY: return 0
        for dx,dy in DIRECTIONS:
            cnt = 1
            for dir_ in (1,-1):
                nx, ny = sx + dx*dir_, sy + dy*dir_
                while in_board(nx,ny) and board[nx,ny]==piece:
                    cnt += 1
                    nx += dx*dir_
                    ny += dy*dir_
            if cnt >= 5:
                return piece
        return 0
    else:
        # 全盘扫描
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x,y]==EMPTY: continue
                piece = board[x,y]
                for dx,dy in DIRECTIONS:
                    cnt = 1
                    nx, ny = x+dx, y+dy
                    while in_board(nx,ny) and board[nx,ny]==piece:
                        cnt += 1
                        nx += dx
                        ny += dy
                    if cnt >= 5:
                        return piece
        return 0

# -------------------------
# 简单启发式评估函数
# -------------------------
# 基本分数表（5连最优）
'''
SCORES = {
    5: 1_000_000,
    4: 10_000,
    3: 1_000,
    2: 100,
    1: 10
}
'''
SCORES = {
    5: 10**9,     # 连五，胜利（相当于无穷大）
    4: 10**7,     # 活四，必须立刻拦
    3: 10**5,     # 活三，能形成杀棋
    2: 10**3,     # 活二
    1: 10         # 单子
}
'''
def evaluate_board(board: np.ndarray, me: int) -> int:
    """
    简单评估：对棋盘上每条连子（所有4个方向）计数并根据是否两端空位（open）加权。
    返回 (me - opp) 的分值（越大越好）
    """
    opp = BLACK if me==WHITE else WHITE
    def score_for(player:int) -> int:
        total = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x,y] != player:
                    continue
                for dx,dy in DIRECTIONS:
                    # 只在连子起点处计数（避免重复）
                    prevx, prevy = x-dx, y-dy
                    if in_board(prevx,prevy) and board[prevx,prevy]==player:
                        continue
                    # 扩展
                    length = 0
                    nx, ny = x, y
                    while in_board(nx,ny) and board[nx,ny]==player:
                        length += 1
                        nx += dx
                        ny += dy
                    end1 = (x-dx, y-dy)
                    end2 = (nx, ny)
                    open_ends = 0
                    if in_board(*end1) and board[end1]==EMPTY:
                        open_ends += 1
                    if in_board(*end2) and board[end2]==EMPTY:
                        open_ends += 1
                    base = SCORES.get(length, 0)
                    if length >=5:
                        total += SCORES[5]
                    else:
                        # open double-ended 加权, 单端次之, 关闭（0）最差
                        if open_ends == 2:
                            total += base * 10
                        elif open_ends == 1:
                            total += base * 2
                        else:
                            total += base
        return total
    return score_for(me) - score_for(opp)
'''



def evaluate_board(board: np.ndarray, me: int) -> int:
    """
    改进评估：必须拦截的局面给极高分
    返回 (me - opp) 的分值
    """
    opp = BLACK if me==WHITE else WHITE

    def score_for(player:int) -> int:
        total = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if board[x,y] != player:
                    continue
                for dx,dy in DIRECTIONS:
                    # 避免重复计数
                    prevx, prevy = x-dx, y-dy
                    if in_board(prevx,prevy) and board[prevx,prevy]==player:
                        continue

                    # 扩展连续棋子长度
                    length = 0
                    nx, ny = x, y
                    while in_board(nx,ny) and board[nx,ny]==player:
                        length += 1
                        nx += dx
                        ny += dy

                    # 两端情况
                    end1 = (x-dx, y-dy)
                    end2 = (nx, ny)
                    open_ends = 0
                    if in_board(*end1) and board[end1]==EMPTY:
                        open_ends += 1
                    if in_board(*end2) and board[end2]==EMPTY:
                        open_ends += 1

                    # 评分逻辑
                    if length >= 5:
                        total += SCORES[5]
                    elif length == 4:
                        if open_ends == 2:
                            total += SCORES[4]  # 活四
                        elif open_ends == 1:
                            total += SCORES[4] // 10  # 冲四
                    elif length == 3:
                        if open_ends == 2:
                            total += SCORES[3]  # 活三
                        elif open_ends == 1:
                            total += SCORES[3] // 10  # 冲三
                    elif length == 2:
                        if open_ends == 2:
                            total += SCORES[2]  # 活二
                        elif open_ends == 1:
                            total += SCORES[2] // 10
                    elif length == 1:
                        total += SCORES[1]
        return total

    return score_for(me) - score_for(opp)

# -------------------------
# 走子生成（启发：只在已有棋子附近的点考虑）
# -------------------------
def generate_moves(board: np.ndarray, radius=3) -> List[Tuple[int,int]]:
    """
    产生候选落子：在现有棋子周围 radius 范围内的空位
    若盘面为空则返回中心点
    """
    occupied = np.argwhere(board != EMPTY)
    if len(occupied)==0:
        mid = BOARD_SIZE//2
        return [(mid,mid)]
    candidates = set()
    for (x,y) in occupied:
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x+dx, y+dy
                if in_board(nx,ny) and board[nx,ny]==EMPTY:
                    candidates.add((nx,ny))
    # 可以按距离中心或启发式排序：这里简单按周围邻子数降序
    def neigh_score(pos):
        x,y = pos
        s=0
        for dx in range(-1,2):
            for dy in range(-1,2):
                nx, ny = x+dx, y+dy
                if in_board(nx,ny) and board[nx,ny] != EMPTY:
                    s += 1
        return -s  # 想要更大的 s 先被选出（所以返回负数便于sorted）
    sorted_moves = sorted(list(candidates), key=neigh_score)
    return sorted_moves

# -------------------------
# Negamax + alpha-beta
# -------------------------
def negamax(board: np.ndarray, depth:int, alpha:int, beta:int, player:int, last_move:Optional[Tuple[int,int]]=None, start_time:Optional[float]=None, time_limit:Optional[float]=None) -> Tuple[int, Optional[Tuple[int,int]]]:
    """
    返回 (best_score, best_move)
    player 指当前走子方 (BLACK or WHITE)
    depth: 搜索深度（叶子用启发式评估）
    支持简单时间限制
    """
    winner = check_win(board, last_move)
    if winner == player:
        return 10**9, None
    elif winner != 0:
        return -10**9, None

    if depth == 0 or (time_limit is not None and time.time()-start_time > time_limit):
        return evaluate_board(board, player), None

    best_score = -math.inf
    best_move = None
    moves = generate_moves(board)
    # 简单 move ordering: 评估每个候选落子并按高分先搜索（one-ply eval）
    scored_moves = []
    for mv in moves:
        x,y = mv
        board[x,y] = player
        sc = evaluate_board(board, player)
        board[x,y] = EMPTY
        scored_moves.append((sc, mv))
    scored_moves.sort(reverse=True, key=lambda t: t[0])

    for _, mv in scored_moves:
        x,y = mv
        board[x,y] = player
        val, _ = negamax(board, depth-1, -beta, -alpha, BLACK if player==WHITE else WHITE, last_move=(x,y), start_time=start_time, time_limit=time_limit)
        val = -val
        board[x,y] = EMPTY
        if val > best_score:
            best_score = val
            best_move = mv
        alpha = max(alpha, val)
        if alpha >= beta:
            break
        if time_limit is not None and time.time()-start_time > time_limit:
            break
    return best_score, best_move

# -------------------------
# 外部接口：AI 下一步
# -------------------------

# 返回分数 落子位置
'''
def ai_best_move(board: np.ndarray, player:int, depth:int=3, time_limit:Optional[float]=None) -> Tuple[int,Tuple[int,int]]:
    start = time.time()
    score, move = negamax(board.copy(), depth, -math.inf, math.inf, player, start_time=start, time_limit=time_limit)
    if move is None:
        # 兜底：随机合法
        moves = generate_moves(board)
        if not moves:
            raise RuntimeError("无合法着法")
        move = moves[0]
    return score, move
'''

def ai_best_move(board: np.ndarray, player:int, depth:int=30, time_limit:float=30.0) -> Tuple[int,Tuple[int,int]]:
    """
    使用迭代加深 + alpha-beta 搜索最佳着法
    max_depth: 最大允许深度（一般 20~30 就够了）
    time_limit: 总思考时间限制（秒）
    """
    start = time.time()
    best_score, best_move = None, None

    for depth in range(1, depth+1):
        remaining = time_limit - (time.time() - start)
        if remaining <= 0:
            break  # 时间到了，停止加深

        score, move = negamax(board.copy(), depth, -math.inf, math.inf,
                              player, start_time=start, time_limit=time_limit)
        if move is not None:
            best_score, best_move = score, move

        # 如果已经找到必胜/必败的局面，没必要再往下搜
        if abs(best_score) >= 10**9:
            break

    if best_move is None:
        # 兜底：找一个合法走子
        moves = generate_moves(board)
        if not moves:
            raise RuntimeError("无合法着法")
        best_move = moves[0]
        best_score = 0

    return best_score, best_move

# -------------------------
# 简单交互（人机对弈）
# -------------------------
def human_vs_ai(start_board:Optional[np.ndarray]=None, human_piece=BLACK, ai_piece=WHITE, ai_depth=20):
    board = start_board.copy() if start_board is not None else np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=int)
    turn = BLACK  # 黑先
    last_move = None
    while True:
        print_board(board)
        winner = check_win(board, last_move)
        if winner != 0:
            print("Winner:", "黑(X)" if winner==BLACK else "白(O)")
            break
        if not np.any(board==EMPTY):
            print("平局")
            break
        if turn == human_piece:
            s = input(f"你是 {'黑(X)' if human_piece==BLACK else '白(O)'}，请输入落子 'x y' (或 'q' 退出): ")
            if s.strip().lower()=='q':
                print("退出")
                break
            try:
                x,y = map(int, s.split())
                if not in_board(x,y) or board[x,y]!=EMPTY:
                    print("无效位置")
                    continue
            except:
                print("格式错误")
                continue
            board[x,y] = human_piece
            last_move = (x,y)
        else:
            print("AI 正在思考...")
            sc, mv = ai_best_move(board, ai_piece, depth=ai_depth, time_limit=20.0)
            x,y = mv
            print(f"AI 下子: {x} {y}  (估值 {sc})")
            board[x,y] = ai_piece
            last_move = (x,y)
        turn = BLACK if turn==WHITE else WHITE


# 文件末尾把“print AI_MOVE”那行删掉或注释掉，改成 return 数据
def ai_decide(board, side, depth=3, time_limit=3.0):
    """供外部调用的纯函数，返回 (x, y, score)"""
    sc, mv = ai_best_move(board, side, depth=depth, time_limit=time_limit)
    return mv[0], mv[1], sc

# -------------------------
# 示例和主程序
# -------------------------

# 如果只需要单步的结果 使用 python gomoku_ai2.py --load board.txt --ai_move --ai_depth 3
# ai默认白色方 我方默认黑色方 导入棋盘进行单步计算时 ai以白色方计算分数
def example_usage():
    # 新棋盘，AI（白）深度3，与人（黑）对弈示例
    board = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=int)
    print("示例：从空盘开始人机对弈（人=黑）")
    human_vs_ai(board, human_piece=BLACK, ai_piece=WHITE, ai_depth=3)

if __name__ == "__main__":
    # 你可以修改这里的调用方式：从txt加载棋盘 / 人机对弈 / AI 计算下一步
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="加载board txt文件路径")
    parser.add_argument("--play", action="store_true", help="进入人机对弈")
    parser.add_argument("--ai_move", action="store_true", help="从加载的棋盘计算AI下一步并输出")
    parser.add_argument("--ai_depth", type=int, default=3)
    parser.add_argument("--ai_side", type=str, help="选择让ai下白棋还是黑棋")
    args = parser.parse_args()

    if args.load:
        b = load_board_from_txt(args.load)
        if args.play:
            human_vs_ai(b, human_piece=BLACK, ai_piece=WHITE, ai_depth=args.ai_depth)
        elif args.ai_move:
            if args.ai_side == 'WHITE':
                sc, mv = ai_best_move(b, WHITE, depth=args.ai_depth, time_limit=3.0) # 默认ai下白棋
                print_board(b)
                print("现在我要下白棋")
                print("AI 估值:", sc, "建议落子:", np.int64(mv))
            elif args.ai_side == 'BLACK':
                sc, mv = ai_best_move(b, BLACK, depth=args.ai_depth, time_limit=3.0) # 默认ai下白棋
                print_board(b)
                print("现在我要下黑棋")
                print("AI 估值:", sc, "建议落子:", np.int64(mv))
        else:
            print_board(b)
    else:
        example_usage()
