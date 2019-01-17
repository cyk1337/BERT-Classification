#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

# @ time    : 2019-01-15 17:09
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : bert_report.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn.metrics import classification_report
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--label', '-l', dest='labelPath', action='store', required=True, help='path to testset')
parser.add_argument('--pred', '-p', dest='predPath', action='store', required=True, help='path to prediction')
parser.add_argument('--output', '-o', dest='outPath', action='store', required=True, help='path to results report')
args = parser.parse_args()

label_path = args.labelPath
pred_path = args.predPath
report_path = args.outPath

# label_path = "/Users/chaiyekun/Documents/CODE/text-classification/data/seg_testset.txt"
# pred_path = "results_dir/test_results.tsv"
labels = ["alerts", "baike", "calculator", "call", "car_limit", "chat", "cook_book", "fm", "general_command",
          "home_command", "master_command", "music", "news", "shopping", "stock", "time", "translator", "video",
          "weather"]

df1 = pd.read_csv(label_path, sep="\t", header=None, names=["X", "y"])
y_true = df1["y"].tolist()
print(len(y_true))

# 1. generate matrix from prob matrix file
y_pred = []
corr_cnt = tot_cnt = 0
with open(pred_path) as f:
    for line in f:
        vec = np.array(line.split(), dtype=np.float32)
        if len(vec) != len(labels):
            continue
        id = np.argmax(vec)
        label = labels[id]
        if y_true[tot_cnt] == label:
            corr_cnt += 1
        y_pred.append(label)
        tot_cnt += 1
print("correct count: " + str(tot_cnt))

# 2. load label file
# print(label_path, pred_path)
# df2 = pd.read_csv(pred_path, header=None, names=["y_pred"])
# y_pred = df2['y_pred'].tolist()

assert len(y_true) == len(y_pred), "# of prediction and labels not match!"
# print(list(zip(y_true, y_pred)))
reports = classification_report(y_true, y_pred, digits=4)
print(reports)
print(reports, file=open(report_path, 'w'))
