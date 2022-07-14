#! /usr/bin/env python
import sklearn
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model import data_helpers
from model.text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pandas as pd


class StructPredictModel(object):
    def __init__(self, model_dir):
        self.checkpoint_dir = model_dir
        self.graph = tf.Graph()
        self.sess = None

    def load_parameter(self, model_name="textcnn"):
        checkpoint_dir = self.checkpoint_dir
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                # saver = tf.train.import_meta_graph(
                #     "{}.meta".format("./runs/1566473224/checkpoints/model-30000"))
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)

    def predict(self, sentence):
        x_raw = sentence
        batch_size = len(x_raw)
        checkpoint_dir = self.checkpoint_dir
        vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x_raw)))

        print("\nEvaluating...\n")

        # Evaluation
        # ==================================================
        # Get the placeholders from the graph by name
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), batch_size, 1, shuffle=False)

        # Collect the predictions here
        predict = []
        for x_test_batch in batches:
            batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            label = ["B", "C", "M", "P", "R"]
            for idx in batch_predictions:
                predict.append(label[idx])
        return predict

    def close(self):
        self.graph.close()
        self.sess.close()