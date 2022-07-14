#! /usr/bin/env python
import sklearn
import tensorflow as tf
import numpy as np
import os
import time
import datetime

from sklearn.metrics import f1_score

from model import data_helpers
from model.text_cnn import TextCNN
from tensorflow.contrib import learn
from tqdm import tqdm
import math
import time

# Parameters
# ==================================================

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "data/train.tsv", "Data source for the train_data_file.")
tf.flags.DEFINE_string("test_data_file", "data/test.tsv", "Data source for the test_data_file.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("fold", 1, "fold")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    labels = data_helpers.load_labels()
    # x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file, FLAGS.test_data_file)
    train_x_text, y_train = data_helpers.load_data_and_labels_new(FLAGS.train_data_file, labels=labels)
    dev_x_text, y_dev = data_helpers.load_data_and_labels_new(FLAGS.test_data_file, labels=labels)
    x_text = train_x_text + dev_x_text

    # Build vocabulary
    print(x_text[:10])
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit(x_text)

    x_train = np.array(list(vocab_processor.transform(train_x_text)))
    x_dev = np.array(list(vocab_processor.transform(dev_x_text)))

    # # Randomly shuffle data
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    #
    # # Split train/test set
    # # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    #
    # del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(FLAGS.fold)))
            if os.path.exists(out_dir):
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(FLAGS.fold) + "_" + timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            logging = data_helpers.get_logger(os.path.join(out_dir, "log"))
            logging.critical("Writing to {}\n".format(out_dir))
            logging.critical(FLAGS.train_data_file)
            logging.critical(FLAGS.test_data_file)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                pbar.set_postfix({
                    'loss': '{0:1.5f}'.format(loss),
                    # "setp": step,
                    # "acc": accuracy,
                })
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
                pbar.update(1)

            def dev_step(x_dev, y_dev, writer=None, best_f1=None):
                """
                Evaluates model on a dev set
                """
                total_dev_correct = 0
                all_l = np.array([], dtype=np.int64)
                all_p = np.array([], dtype=np.int64)
                dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                for dev_batch in dev_batches:
                    x_batch, y_batch = zip(*dev_batch)
                    feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, correct_labels, predictions = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.correct_labels, cnn.predictions],
                        feed_dict)
                    all_l = np.concatenate((all_l, correct_labels))
                    all_p = np.concatenate((all_p, predictions))
                f1 = f1_score(all_l, all_p, average="weighted")
                logging.critical(sklearn.metrics.classification_report(all_l, all_p, digits=4, target_names=labels))
                if f1 > best_f1:
                    # save model
                    best_f1 = f1
                    current_step = tf.train.global_step(sess, global_step)
                    path = saver.save(sess, checkpoint_prefix)
                    logging.critical("Saved best model checkpoint to {}\n".format(path))
                return best_f1
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # if writer:
                #     writer.add_summary(summaries, step)

            # Training loop. For each batch...
            labels = ["B", "C", "M", "P", "R"]
            best_f1 = 0.
            for epoch in range(FLAGS.num_epochs):
                # Generate batches
                nbatches = (len(x_train) + FLAGS.batch_size - 1) // FLAGS.batch_size
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                pbar = tqdm(total=nbatches, desc="epoch {}".format(epoch))
                for batch in tqdm(batches):
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    # if current_step % FLAGS.evaluate_every == 0:
                    #     print("\nEvaluation:")
                    #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    #     print("")
                    # if current_step % FLAGS.checkpoint_every == 0:
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))
                pbar.close()
                # test performance
                best_f1 = dev_step(x_dev, y_dev, writer=None, best_f1=best_f1)



def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
