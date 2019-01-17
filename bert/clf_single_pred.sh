#!/usr/bin/env bash

export BERT_BASE_DIR=$1
export inputX=$2
BASEDIR=$3
SAVE_DIRNAME=$4
global_steps=$5
MaxLen=$6

#source activate cyk
source activate /export/cyk/envs

python ${BASEDIR}/bert/run_classification.py \
  --task_name=domainclf \
  --do_train=false \
  --do_eval=false \
  --do_predict=false \
  --do_single_predict=true \
  --X_input=$inputX \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=${BASEDIR}/${SAVE_DIRNAME}/model.ckpt-${global_steps} \
  --max_seq_length=$MaxLen \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0