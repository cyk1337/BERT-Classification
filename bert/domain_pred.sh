#!/usr/bin/env bash

export BERT_BASE_DIR=chinese_L-12_H-768_A-12
export DATA_DIR=../data
export TRAINED_CLASSIFIER=tmp

python run_classification.py \
  --task_name=domainclf \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATA_DIR\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=30 \
  --output_dir=tmp/test_output/