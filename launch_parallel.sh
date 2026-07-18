#!/bin/bash
# launch_parallel.sh
#
# 并行启动多个 Fair driver（一个 script 一个 tmux window + 独立 log）
#
# 用法:
#   bash launch_parallel.sh Fair_attack_cifar10.py Fair_cgba_cifar10.py Fair_cgbah_cifar10.py
#   bash launch_parallel.sh Fair_attack_imagenet.py
#
# 每个 script 会：
#   - 在独立 tmux session 中跑 (session 名 = 文件名去掉 .py)
#   - stdout/stderr 存到 logs/<session_name>.log
#
# 监控:
#   tmux ls                        # 列出所有 session
#   tmux attach -t <session_name>  # 进入某个 session (Ctrl+B, D 退出)
#   tail -f logs/*.log             # 实时看所有 log

set -e

LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <script1.py> [script2.py] [script3.py] ..."
    echo ""
    echo "Example:"
    echo "  bash $0 Fair_attack_cifar10.py Fair_cgba_cifar10.py Fair_cgbah_cifar10.py Fair_Surfree_cifar10.py"
    exit 1
fi

echo "── Launching ${#} scripts in parallel ─────────"

for script in "$@"; do
    if [ ! -f "$script" ]; then
        echo "  ✗ File not found: $script (skip)"
        continue
    fi

    # session 名 = 文件名 (去 .py)
    session_name=$(basename "$script" .py)
    log_file="$LOG_DIR/${session_name}.log"

    # 已经在跑就跳过
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "  ⚠ [$session_name] 已在跑 (tmux session 存在)，跳过"
        continue
    fi

    tmux new-session -d -s "$session_name" \
        "python $script 2>&1 | tee $log_file"
    echo "  ✓ [$session_name] → $log_file"
done

echo ""
echo "所有 session:"
tmux ls 2>/dev/null || echo "(无)"

echo ""
echo "查看某个 session:  tmux attach -t <name>   (Ctrl+B, D 退出)"
echo "实时看 log:        tail -f $LOG_DIR/*.log"
echo "杀掉所有 session:  tmux kill-server"
