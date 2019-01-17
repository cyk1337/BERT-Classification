#!/usr/bin/env bash

export BERT_BASE_DIR=$1
export DATA_DIR=$2
BASEDIR=$3
SAVE_DIRNAME=$4
global_steps=$5
MaxLen=$6

#source activate cyk
source activate /export/cyk/envs

echo "python ${BASEDIR}/bert/run_classification.py \
  --task_name=domainclf \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=${DATA_DIR} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=${BASEDIR}/${SAVE_DIRNAME}_maxLen${MaxLen}/model.ckpt-${global_steps} \
  --max_seq_length=$MaxLen \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=${BASEDIR}/${SAVE_DIRNAME}/result"

python ${BASEDIR}/bert/run_classification.py \
  --task_name=domainclf \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=${BASEDIR}/${SAVE_DIRNAME}/model.ckpt-${global_steps} \
  --max_seq_length=$MaxLen \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=${BASEDIR}/${SAVE_DIRNAME}/result