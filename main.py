#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

# @ time    : 2019-01-14 16:26
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : main.py

"""

import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="training or test mode", default="train")
parser.add_argument("-s", "--save_dirname", help="SaveDirName", default="BERT00")
parser.add_argument("-gs", "--global_step", help="global steps")
parser.add_argument("-X", "--inputX", help="single X to be predicted")
args = parser.parse_args()
mode = args.mode
save_dirname = args.save_dirname
global_step = args.global_step
inputX = args.inputX

basedir = os.path.dirname(os.path.abspath(__file__))

print("Current dir:", basedir)

config_path = os.path.join(basedir, 'config.yaml')
print(config_path)

f = open(config_path, encoding='utf8')
config_settings = yaml.load(f.read())
f.close()

data_settings = config_settings["Data"]

results_dir = os.path.join(basedir, data_settings["results_dir"])

data_dir = os.path.join(basedir, data_settings["data_dir"])

raw_trainset = os.path.join(data_dir, data_settings["raw_train_data"])
raw_testset = os.path.join(data_dir, data_settings["raw_test_data"])

processed_trainset = os.path.join(data_dir, data_settings["processed_trainset"])
processed_testset = os.path.join(data_dir, data_settings["processed_testset"])

ft_seg_trainset = os.path.join(data_dir, data_settings["segmented_trainset"])
ft_seg_testset = os.path.join(data_dir, data_settings["segmented_testset"])

ft_unseg_trainset = os.path.join(data_dir, data_settings["ft_unseg_trainset"])
ft_unseg_testset = os.path.join(data_dir, data_settings["ft_unseg_testset"])

seg_split_trainset = os.path.join(data_dir, data_settings["seg_split_trainset"])
seg_split_devset = os.path.join(data_dir, data_settings["seg_split_devset"])



def BERT():
    #                 BERT usage
    # ================================================================= #
    # train:
    #           python main.py -m "train" -s BERT0
    # output redirection -> python main.py -m "train" -s BERT0  >  out100.txt  2>&1  &
    # test:
    #           python main.py -m "test" -s BERT0 -gs 22729
    # predict:
    #           python main.py -m "predict" -s BERT0 -gs 22729 -X "abc"
    # ================================================================= #
    BERT_DIR = os.path.join(basedir, "bert")
    BERT_BASE_DIR = os.path.join(BERT_DIR, "chinese_L-12_H-768_A-12")

    # Bert parameters
    maxLen = 30

    if mode == "train":
        cmd = "sh bert/{} {} {} {} {} {}".format("clf_domain.sh", BERT_BASE_DIR, data_dir, basedir, save_dirname,
                                                 maxLen)
        print(cmd)
        os.system(cmd)

    elif mode == "test":
        res_dir = os.path.join(basedir, save_dirname, 'result')
        prob_file = os.path.join(res_dir, "test_results.tsv")
        report_file = os.path.join(res_dir, "report.txt")

        # os.system(
        #     "sh bert/{} {} {} {} {} {}".format("clf_predict.sh", BERT_BASE_DIR, data_dir, basedir, save_dirname,
        #                                        global_step))
        # ========= past solution =========

        if not os.path.exists(prob_file):
            os.system(
                "sh bert/{} {} {} {} {} {} {}".format("clf_predict.sh", BERT_BASE_DIR, data_dir, basedir, save_dirname,
                                                      global_step, maxLen))
        else:
            print('-' * 100)
            print("{} exists!".format(prob_file))
        clac_per_class = "python {}/bert_report.py -l {} -p {} -o {}".format(BERT_DIR, raw_testset, prob_file,
                                                                             report_file)
        print(clac_per_class)
        os.system(clac_per_class)
        # ========= past solution =========

    elif mode == "predict":
        os.system(
            "sh bert/{} {} {} {} {} {} {}".format("clf_single_pred.sh", BERT_BASE_DIR, inputX, basedir, save_dirname,
                                                  global_step, maxLen))


if __name__ == '__main__':
    #fastText()
    BERT()
