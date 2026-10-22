#!/usr/bin/env bash
# 串行运行 U1652 两个方向，以及 SUES-200 四个高度的两个方向。
# 先准备好两个数据集；DRY_RUN=1 时仅打印命令。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-24}
BATCH_SIZE_EVAL=${BATCH_SIZE_EVAL:-128}
GPU_ID=${GPU_ID:-0}
U1652_DATA=${U1652_DATA:-./data/U1652}
SUES200_DATA=${SUES200_DATA:-./data/SUES-200}
OUTPUT_ROOT=${OUTPUT_ROOT:-./checkpoints/all}
DRY_RUN=${DRY_RUN:-0}

# 日期和进程号用于区分批次；每个检索方向也使用独立目录。
run_root="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$$"
common_args=(
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --batch_size_eval "$BATCH_SIZE_EVAL"
    --device "cuda:$GPU_ID"
    --gpu_ids "$GPU_ID"
)

require_dir() {
    if [[ ! -d "$1" ]]; then
        printf '缺少数据目录：%s\n请先整理数据，再启动全部实验。\n' "$1" >&2
        exit 1
    fi
}

if [[ "$DRY_RUN" != 1 ]]; then
    # 启动前检查全部高度和方向，避免训练到中途才发现目录缺失。
    for view in satellite drone; do
        require_dir "$U1652_DATA/train/$view"
    done
    for view in query_drone query_satellite gallery_drone gallery_satellite; do
        require_dir "$U1652_DATA/test/$view"
    done
    for altitude in 150 200 250 300; do
        for view in satellite drone; do
            require_dir "$SUES200_DATA/Training/$altitude/$view"
        done
        for view in query_drone query_satellite gallery_drone gallery_satellite; do
            require_dir "$SUES200_DATA/Testing/$altitude/$view"
        done
    done
    mkdir -p -- "$run_root"
fi

run_experiment() {
    local name=$1
    shift
    printf '\n运行 %s\n' "$name"
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then
        return
    fi
    # 保存完整启动命令及 stdout/stderr；任一训练失败即停止批次。
    printf '%q ' "$@" > "$run_root/$name.command.txt"
    printf '\n' >> "$run_root/$name.command.txt"
    "$@" 2>&1 | tee "$run_root/$name.console.log"
}

for direction in D2S S2D; do
    run_experiment "U1652-$direction" \
        "$PYTHON" -u train_university.py "${common_args[@]}" \
        --dataset "U1652-$direction" \
        --data_folder "$U1652_DATA" \
        --model_path "$run_root/U1652/$direction"
done

for altitude in 150 200 250 300; do
    for direction in D2S S2D; do
        run_experiment "SUES200-$altitude-$direction" \
            "$PYTHON" -u train_sues200.py "${common_args[@]}" \
            --dataset "SUES200-$direction" \
            --altitude "$altitude" \
            --data_folder "$SUES200_DATA" \
            --model_path "$run_root/SUES200"
    done
done

if [[ "$DRY_RUN" == 1 ]]; then
    printf '\n预览完成，共 10 组训练及评估；未启动训练。\n'
else
    printf '\n全部 10 组训练及评估完成，结果目录：%s\n' "$run_root"
fi
