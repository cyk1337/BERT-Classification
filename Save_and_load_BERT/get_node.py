#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

# @ time    : 2019-01-22 18:08
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : get_node.py

"""

import tensorflow as tf
from google.protobuf import text_format


def get_all_nodes(pbtxt_file):
    with open(pbtxt_file) as f:
        graph_def = text_format.Parse(f.read(), tf.GraphDef())

    nodes = [n.name for n in graph_def.node]
    for n in nodes:
        print(n, file=open("nodes.txt",'a'))
        print(n)


def get_in_out_nodes_name(pb_path):
    gf = tf.GraphDef()
    gf.ParseFromString(open(pb_path, "rb").read())
    print([n.name + '=>' + n.op for n in gf.node if n.op in ('Softmax', 'Placeholder')])
    # retrained graph
    print([n.name + '=>' + n.op for n in gf.node if n.op in ('Softmax', 'Mul')])


if __name__ == '__main__':
    pbtxt_path = './graph.pbtxt'
    get_all_nodes(pbtxt_path)

    pb_path = "frozen_model.pb"
    # get_in_out_nodes_name(pb_path)