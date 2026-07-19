#!/bin/bash
# launch_parallel.sh
#
# 生成 M × N 个 driver 副本（每个 attack × 每个 model 一个 .py 文件）
# 不启动任何东西，只是复制 + sed 编辑
#
# 用法（用 -- 分隔 drivers 和 models）:
#   bash launch_parallel.sh <driver1> [driver2] ... -- <model1> [model2] ...
#
# 示例 (3 attack × 3 model = 9 文件):
#   bash launch_parallel.sh \
#     Fair_attack_cifar10.py Fair_cgba_cifar10.py Fair_cgbah_cifar10.py \
#     -- preactresnet18 wideresnet40_2 vit

set -e

# 分离 drivers 和 models（用 -- 分隔）
drivers=()
models=()
found_sep=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        found_sep=true
        continue
    fi
    if $found_sep; then
        models+=("$arg")
    else
        drivers+=("$arg")
    fi
done

if [ ${#drivers[@]} -eq 0 ] || [ ${#models[@]} -eq 0 ]; then
    echo "Usage: bash $0 <driver1> [driver2] ... -- <model1> [model2] ..."
    echo ""
    echo "示例:"
    echo "  bash $0 Fair_attack_cifar10.py Fair_cgba_cifar10.py Fair_cgbah_cifar10.py \\"
    echo "         -- preactresnet18 wideresnet40_2 vit"
    exit 1
fi

echo "── 生成 ${#drivers[@]} attack × ${#models[@]} model = $((${#drivers[@]} * ${#models[@]})) 个文件 ────────"

for driver in "${drivers[@]}"; do
    if [ ! -f "$driver" ]; then
        echo "  ✗ Driver not found: $driver (skip)"
        continue
    fi
    base=$(basename "$driver" .py)

    for model in "${models[@]}"; do
        copy_name="${base}_${model}.py"
        cp "$driver" "$copy_name"

        sed -i "s/^MODEL_NAMES\s*=.*/MODEL_NAMES = ['$model']/" "$copy_name"
        sed -i "s/^model_arc\s*=.*/model_arc = '$model'/" "$copy_name"

        echo "  ✓ $copy_name"
    done
done

echo ""
echo "生成完毕。手动启动 python 即可。"
