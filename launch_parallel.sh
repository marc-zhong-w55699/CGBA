#!/bin/bash
# launch_parallel.sh
#
# 一个 attack driver × N 个 model 并行跑
# 自动复制 driver，改 MODEL_NAMES 或 model_arc，独立 tmux + log
#
# 用法:
#   bash launch_parallel.sh <driver.py> <model1> <model2> [model3] ...
#
# 示例 (CIFAR M2D on 3 models):
#   bash launch_parallel.sh Fair_attack_cifar10.py preactresnet18 wideresnet40_2 vit
#
# 示例 (CIFAR CGBA on 3 models):
#   bash launch_parallel.sh Fair_cgba_cifar10.py preactresnet18 wideresnet40_2 vit
#
# 示例 (ImageNet M2D on 4 models):
#   bash launch_parallel.sh Fair_attack_imagenet.py resnet50 vgg19 inception_v3 ViT
#
# 监控:
#   tmux ls
#   tmux attach -t <session>       (Ctrl+B, D 退出)
#   tail -f /root/autodl-tmp/logs/*.log
#
# 停止:
#   tmux kill-session -t <session>
#   tmux kill-server               # 全停

set -e

LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"

if [ $# -lt 2 ]; then
    echo "Usage: bash $0 <driver.py> <model1> [model2] [model3] ..."
    echo ""
    echo "示例:"
    echo "  bash $0 Fair_attack_cifar10.py preactresnet18 wideresnet40_2 vit"
    echo "  bash $0 Fair_cgba_imagenet.py resnet50 vgg19 inception_v3 ViT"
    exit 1
fi

driver="$1"
shift
models=("$@")

if [ ! -f "$driver" ]; then
    echo "✗ Driver not found: $driver"
    exit 1
fi

# 去 .py 后缀作为 base name
base=$(basename "$driver" .py)

echo "── 并行启动 [$driver] × ${#models[@]} models ──────────"

for model in "${models[@]}"; do
    # 复制 driver: Fair_attack_cifar10.py → Fair_attack_cifar10_<model>.py
    copy_name="${base}_${model}.py"
    cp "$driver" "$copy_name"

    # 用 sed 改 MODEL_NAMES (CIFAR) 或 model_arc (ImageNet)
    # 两条 sed 都跑，只影响存在的那种模式
    sed -i "s/^MODEL_NAMES\s*=.*/MODEL_NAMES = ['$model']/" "$copy_name"
    sed -i "s/^model_arc\s*=.*/model_arc = '$model'/" "$copy_name"

    session_name="${base}_${model}"
    log_file="$LOG_DIR/${session_name}.log"

    # 如果已在跑就先 kill
    tmux kill-session -t "$session_name" 2>/dev/null || true

    tmux new-session -d -s "$session_name" \
        "python $copy_name 2>&1 | tee $log_file"

    echo "  ✓ [$session_name] → $log_file"
done

echo ""
echo "所有 session:"
tmux ls 2>/dev/null || echo "(无)"

echo ""
echo "查看 log:      tail -f $LOG_DIR/*.log"
echo "进入 session:  tmux attach -t <name>   (Ctrl+B, D 退出)"
echo "全部停止:      tmux kill-server"
