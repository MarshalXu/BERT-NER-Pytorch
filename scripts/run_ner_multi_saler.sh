#!/bin/bash

# 获取当前目录
CURRENT_DIR=$(pwd)
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-multi-cased
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="yiliang_multigual"




# 检查命令行参数
if [ "$1" = "train" ]; then
    export WANDB_MODE="online"  # 训练模式使用 online
    DO_TRAIN="--do_train --do_eval"
    DO_PREDICT=""
    # 获取当前日期和时间
    current_time=$(date +"%Y-%m-%d_%H-%M-%S")
    # 创建运行名称
    run_name="run_$TASK_NAME""_$current_time"
    # 设置 WANDB_NAME 环境变量
    export WANDB_NAME=$run_name
    # 输出当前设置的运行名称，确认它已被正确设置
    echo "Current WANDB run name is set to: $WANDB_NAME"
    
elif [ "$1" = "predict" ]; then
    export WANDB_MODE="disabled"  # 预测模式禁用wandb
    DO_TRAIN=""
    DO_PREDICT="--do_predict --do_eval"
else
    echo "Invalid argument: Please use 'train' or 'predict'."
    exit 1
fi

# 执行Python脚本
python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  $DO_TRAIN \
  $DO_PREDICT \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --crf_learning_rate=1e-3 \
  --num_train_epochs=4.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 \
  --loss_type=ce \
  --warmup_proportion=0.05