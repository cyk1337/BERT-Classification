#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

# @ time    : 2019-01-22 18:40
# @ author  : Yekun CHAI
# @ email   : chaiyekun@gmail.com
# @ file    : save_pb.py

"""
import tensorflow as tf
from tensorflow import graph_util
import tokenization
import os
import numpy as np

max_seq_length = 128
batch_size = 1
label_list = ["alerts", "baike", "calculator", "call", "car_limit", "chat", "cook_book", "fm", "general_command",
              "home_command", "master_command", "music", "news", "shopping", "stock", "time", "translator", "video",
              "weather"]


def get_labels():
    """get class."""
    return ["alerts", "baike", "calculator", "call", "car_limit", "chat", "cook_book", "fm", "general_command",
            "home_command", "master_command", "music", "news", "shopping", "stock", "time", "translator", "video",
            "weather"]


class InputFeature:
    def __init__(self, input_ids, input_mask, seg_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seg_ids = seg_ids


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint: init checkpoint save path
    :param output_graph: pb model save path
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "loss/Softmax"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        ## BERT Model requires to load the `create_model` function for the session

        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
        print(output_graph_def.node)

        # for op in graph.get_operations():
        #     print(op.name, op.values())


def tf_pb_predict(pb_path, feat):
    '''
    :param pb_path:pb file path
    :param feat: input feat
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # input tensor name
            input_ids = sess.graph.get_tensor_by_name("input_ids:0")
            input_mask = sess.graph.get_tensor_by_name("input_mask:0")
            seg_ids = sess.graph.get_tensor_by_name("segment_ids:0")

            # output tesor name
            output_tensor_name = sess.graph.get_tensor_by_name("loss/Softmax:0")

            input_ids__ = np.reshape([feat.input_ids], [1, max_seq_length])
            input_mask__ = np.reshape([feat.input_mask], [1, max_seq_length])
            # input_seg_ids__ = np.reshape([feat.seg_ids], [max_seq_length])
            out = sess.run(output_tensor_name, feed_dict={input_ids: input_ids__,
                                                          input_mask: input_mask__,
                                                          seg_ids: feat.seg_ids})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            label_id = sess.run(tf.argmax(score, 1))[0]
            # print(label_id)
            label = label_list[label_id]
            print("pre class_id:{}, label: {}".format(label_id, label))
            return out


def process_unsgetext(text: str, vocab_file, do_lower_case=True):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    tokens_ = tokenizer.tokenize(text)
    if len(text) + 2 > max_seq_length:
        tokens_ = tokens_[:max_seq_length - 2]
    tokens = ["[CLS]"] + tokens_ + ["[SEP]"]
    n = len(tokens)
    seg_ids = [0] * n
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * n
    if n < max_seq_length:
        seg_ids = seg_ids + [0] * (max_seq_length - n)
        input_ids = input_ids + [0] * (max_seq_length - n)
        input_mask = input_mask + [0] * (max_seq_length - n)

    assert len(seg_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return InputFeature(input_ids, input_mask, seg_ids)


def predict_single_case(text, pb_path, vocab_file):
    feat = process_unsgetext(text, vocab_file)
    tf_pb_predict(pb_path, feat)


def run_bert():
    # path of ckpt file
    input_checkpoint = 'model.ckpt-22729'
    #  output path of pb model
    out_pb_path = "bert_pb.pb"
    vocab_file = "vocab.txt"

    testX = "现在PM25怎么样"

    # 调用freeze_graph将ckpt转为pb (BERT model需要修改！！！ 先create_model)
    # if not os.path.exists(out_pb_path):
    #     freeze_graph(input_checkpoint, out_pb_path)

    predict_single_case(testX, out_pb_path, vocab_file)


if __name__ == '__main__':
    run_bert()
