# main.py
import sys
from gomoku_ai2 import load_board_from_txt
from gomoku_ai2 import ai_decide
from connector import TCPConnector
import argparse
from server import TCPServer
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True, help="棋盘文件")
    parser.add_argument("--side", choices=["BLACK", "WHITE"], required=True)
    parser.add_argument("--ip",   default="192.168.1.100", help="机器人 IP")
    parser.add_argument("--port", type=int, default=9900)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    board = load_board_from_txt(args.load)
    side = 1 if args.side == "BLACK" else 2
    depth = args.depth

    # 1. AI 计算
    x, y, score = ai_decide(board, side, depth, 20.0)
    print(f"[AI] 落子: ({x},{y})  分数: {score}")

    # 2. 通过网络发给机器人
    with TCPConnector(args.ip, args.port) as conn:
        conn.send_move(x, y, score)
        # 可选：等待机器人回传“到达”信号
        echo = conn.recv_echo()
        print(f"[Robot Echo] {echo}")
'''

'''

使用方法：
python main.py --load [棋盘文件] --side [WHITE, BLACK] 
可选：--port[端口，默认9003] --depth[搜索深度，默认3]

'''
def main():
    # server = TCPServer(port=9003)
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True, help="棋盘文件")
    parser.add_argument("--side", choices=["BLACK", "WHITE"], required=True)
    # parser.add_argument("--ip",   default="192.168.1.100", help="机器人 IP")
    parser.add_argument("--port", type=int, default=9003)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    server = TCPServer(port=args.port)

    board = load_board_from_txt(args.load)
    side = 1 if args.side == "BLACK" else 2
    depth = args.depth

    # 1. AI 计算
    x, y, score = ai_decide(board, side, depth, 20.0)
    print(f"[AI] 落子颜色：({args.side}) 落子: ({x},{y})  分数: {score}")

    try:
        while True:
            # 示例：上位机根据策略发送决策
            # x, y, score = 3, 4, 100
            x, y, score = (6, 8, 0)
            server.send_move(x, y)

            # 等待机器人反馈状态
            status = server.recv_status()
            if status:
                print(f"[上位机] 机器人状态反馈: {status}")

    except KeyboardInterrupt:
        print("\n[Server] 手动退出。")
    finally:
        server.close()

if __name__ == "__main__":
    main()
