# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import groupby, cycle

import json
import subprocess
import PIL
import coremltools
import math
import os
import tensorflow as tf
import tfcoreml
from PIL import Image
import cv2
import numpy as np

from datasets import dataset_factory
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.font_manager import FontManager
from nets import nets_factory
from operator import itemgetter
from preprocessing import preprocessing_factory
from datasets import dataset_utils
from datasets.plants import read_label_file
from nets import resnet_v2
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.python.training import monitored_session
import seaborn as sns
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

OUTPUT_MODEL_NODE_NAMES_DICT = {
    'resnet_v2_50': 'resnet_v2_50/predictions/Reshape_1',
    'mobilenet_v1': 'MobilenetV1/Predictions/Reshape_1',
}

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

CHECKPOINT_PATH = '/tmp/tfmodel/'
# CHECKPONT_PATH = 'resnet_v2_50_plants_non_exif'
# CHECKPONT_PATH = 'resnet_v2_50_plants_0426'
CHECKPOINT_PATH = 'resnet_v2_50_plants_0617'
CHECKPOINT_PATH = 'experiments/mobilenet_v1_plants_0620'
MODEL_DIR = os.environ.get('MODEL_DIR') or CHECKPOINT_PATH
tf.app.flags.DEFINE_string(
    'checkpoint_path', CHECKPOINT_PATH,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'plants', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

DATASET_DIR = '/projects/private/plant/data_non_exif'
tf.app.flags.DEFINE_string(
    'dataset_dir', DATASET_DIR,
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

# model_name = 'resnet_v2_50'
model_name = 'mobilenet_v1'
tf.app.flags.DEFINE_string(
    'model_name', model_name, 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None,
    'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def inspect_tfrecords(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    examples = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        examples.append(example)
        # print(example)

    return examples


def get_info(checkpoint_path=None,
             calculate_confusion_matrix=False):
    # if not FLAGS.dataset_dir:
    #   raise ValueError('You must supply the dataset directory with --dataset_dir')

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.Graph().as_default()
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    num_classes = (dataset.num_classes - FLAGS.labels_offset)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_epochs=1,  # 每張只讀一次
        # num_readers=1,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    # common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset
    raw_images = image

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        allow_smaller_final_batch=True,
        capacity=5 * FLAGS.batch_size)

    one_hot_labels = slim.one_hot_encoding(
        labels, dataset.num_classes - FLAGS.labels_offset)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            slim.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    if calculate_confusion_matrix:
        confusion_matrix = tf.confusion_matrix(labels=labels,
                                               num_classes=num_classes,
                                               predictions=predictions)
    else:
        confusion_matrix = None

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
        num_batches = FLAGS.max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    checkpoint_path = checkpoint_path or FLAGS.checkpoint_path
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    tf.logging.info('Evaluating %s' % checkpoint_path)
    labels_to_names = read_label_file(FLAGS.dataset_dir)
    probabilities = tf.nn.softmax(logits)
    softmax_cross_entropy_loss = slim.losses.softmax_cross_entropy(
        logits, one_hot_labels, label_smoothing=0.0, weights=1.0)
    grad_imgs = tf.gradients(softmax_cross_entropy_loss,
                             images)[:][0]

    return {
        'labels_to_names': labels_to_names,
        'checkpoint_path': checkpoint_path,
        'num_batches': num_batches,
        'names_to_values': names_to_values,
        'names_to_updates': names_to_updates,
        'variables_to_restore': variables_to_restore,
        'images': images,
        'raw_images': raw_images,
        'network_fn': network_fn,
        'labels': labels,
        'logits': logits,
        'probabilities': probabilities,
        'predictions': predictions,
        'confusion_matrix': confusion_matrix,
        'loss': softmax_cross_entropy_loss,
        'grad_imgs': grad_imgs,
    }


def get_monitored_session(checkpoint_path):
    session_creator = monitored_session.ChiefSessionCreator(
        checkpoint_filename_with_path=checkpoint_path,
        # scaffold=scaffold,
        # master=master,
        # config=config
    )
    return monitored_session.MonitoredSession(
        session_creator=session_creator)


def plot_confusion_matrix(confusion_matrix, labels_to_names=None,
                          save_dir='.'):
    set_matplot_zh_font()
    # ax = plt.subplot()
    fig, ax = plt.subplots()
    # the size of A4 paper
    fig.set_size_inches(18, 15)

    # https://stackoverflow.com/questions/22548813/python-color-map-but-with-all-zero-values-mapped-to-black
    # confusion_matrix = np.ma.masked_where(confusion_matrix < 0.01,
    #                                       confusion_matrix)
    cmap = plt.get_cmap('Accent')
    # cmap = plt.get_cmap('coolwarm')
    # cmap = plt.get_cmap('plasma')
    # cmap = plt.get_cmap('Blues')
    # cmap.set_bad(color='black')

    mask = np.zeros_like(confusion_matrix)
    mask[confusion_matrix == 0] = True
    # sns.set(font_scale=1)
    with sns.axes_style('darkgrid'):
        sns.heatmap(confusion_matrix,
                    linewidths=0.2,
                    linecolor='#eeeeee',
                    xticklabels=True,
                    yticklabels=True,
                    mask=mask, annot=False, ax=ax, cmap=cmap)
    n = confusion_matrix.shape[0]

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    axis = [labels_to_names[i] if labels_to_names else i
            for i in range(n)]
    ax.xaxis.set_ticklabels(axis, rotation=270)
    ax.yaxis.set_ticklabels(axis, rotation=0)

    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    print('plot shown')
    plt.show()


def get_matplot_zh_font():
    # From https://blog.csdn.net/kesalin/article/details/71214038
    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)

    output = subprocess.check_output('fc-list :lang=zh-tw -f "%{family}\n"',
                                     shell=True)
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = list(mat_fonts & zh_fonts)

    return available


def set_matplot_zh_font():
    available = get_matplot_zh_font()
    if len(available) > 0:
        plt.rcParams['font.sans-serif'] = [available[0]]  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False


def deprocess_image(x, target_std=0.15):
    epsilon = 1e-7
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + epsilon)
    x *= target_std

    # clip to [0, 1]
    # x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # if K.image_data_format() == 'channels_first':
    #     x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def plot_silency(silency, image, file_name=None):
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silency)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    if file_name:
        plt.savefig(file_name)
        print(file_name, 'saved')
    else:
        print('plot shown')
        plt.show()


def _run_info():
    calculate_confusion_matrix = False
    info = get_info(calculate_confusion_matrix=calculate_confusion_matrix)
    checkpoint_path = info['checkpoint_path']
    checkpoint_dir_path = os.path.dirname(checkpoint_path)
    num_batches = info['num_batches']
    names_to_updates = info['names_to_updates']
    variables_to_restore = info['variables_to_restore']
    labels_to_names = info['labels_to_names']
    feed_dict = {}
    y, _ = info['network_fn'](info['images'], reuse=True)

    all_predictions = []
    all_labels = []
    all_logits = []
    all_probabilities = []
    num_categories = len(info['labels_to_names'])
    all_confusion_matrix = None

    # all_confusion_matrix = np.loadtxt('confusion_matrix.txt')
    # plot_confusion_matrix(all_confusion_matrix,
    #                       labels_to_names=info['labels_to_names'])
    # return
    confusion_matrix_txt = os.path.join(checkpoint_dir_path,
                                        'confusion_matrix.txt')
    with get_monitored_session(checkpoint_path) as sess:
        for i in range(int(math.ceil(num_batches))):
            # break
            print('batch #{} of {}'.format(i, num_batches))
            params = {
                k: v
                for k, v in info.items()
                if isinstance(v, tf.Tensor)
                   and k in [
                       'labels',
                       'images',
                       # 'raw_images',
                       'logits',
                       'probabilities',
                       'predictions',
                       'confusion_matrix',
                       # 'loss',
                       'grad_imgs',
                   ]
            }
            # params.update(
            #     y=y,
            # )
            # params = {'labels': info['labels']}
            try:
                res = sess.run(params, feed_dict=feed_dict)
            except:
                import traceback
                traceback.print_exc()
                raise
                # break

            images = res['images']
            grad_imgs = res['grad_imgs']

            n = images.shape[0]
            save_dir = 'silency_maps'
            try:
                os.makedirs(save_dir)
            except OSError:
                pass

            for j in range(n):
                image = images[j]
                grad_img = grad_imgs[j]
                file_name = '{}/{:03d}_{:03d}.jpg'.format(save_dir, i, j)
                silency = deprocess_image(grad_img)
                plot_silency(silency, image, file_name=file_name)

            labels = res['labels']
            # print(labels)
            # print(res.keys())
            predictions = res['predictions']
            probabilities = res['probabilities']
            logits = res['logits']
            # print(predictions)
            print('len labels', len(labels))
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_logits.extend(logits)
            all_probabilities.extend(probabilities)
            print('all_labels length', len(all_labels))
            print('all_labels unique length', len(set(all_labels)))
            # continue

            # print([x == y for x, y, in zip(predictions, labels)])
            if calculate_confusion_matrix:
                confusion_matrix = res['confusion_matrix']
                print('confusion_matrix')
                print(confusion_matrix)
                if all_confusion_matrix is None:
                    all_confusion_matrix = np.matrix(confusion_matrix)
                else:
                    all_confusion_matrix += np.matrix(confusion_matrix)

                print('all_confusion_matrix')
                print(all_confusion_matrix)
                np.savetxt(confusion_matrix_txt, all_confusion_matrix,
                           fmt='% 3d')
                # if i > 0:
                #     break

        from collections import Counter
        c = Counter(all_labels)
        kv_pairs = sorted(dict(c).items(), key=lambda p: p[0])
        for k, v in kv_pairs:
            print(k, v)

        all_confusion_matrix = np.loadtxt(confusion_matrix_txt)

        info_file_path = os.path.join(checkpoint_dir_path, 'info.json')
        with open(info_file_path, 'w') as f:
            json.dump({
                'labels': all_labels,
                'predictions': all_predictions,
                'probabilities': [l.tolist() for l in all_probabilities],
                'logits': [l.tolist() for l in all_logits],
                'confusion_matrix': all_confusion_matrix.tolist(),
            }, f)

        ranking = _get_accuracy_ranking(all_confusion_matrix, labels_to_names)
        for name, accuracy_ in ranking:
            print(name, accuracy_)

            # plot_confusion_matrix(all_confusion_matrix,
            #                       labels_to_names=labels_to_names,
            #                       save_dir=checkpoint_dir_path)


def _plot_roc(logits_list, labels, predictions, probabilities):
    possible_labels = list(sorted(set(labels)))
    y_binary = label_binarize(labels, classes=possible_labels)

    output_matrix = np.array(probabilities)
    y_score_matrix = output_matrix
    y_score_matrix = np.where(
        y_score_matrix == np.max(y_score_matrix, axis=1)[:, None],
        y_score_matrix, 0)

    tpr = {}
    fpr = {}
    roc_auc = {}
    for i in range(len(possible_labels)):
        y_scores = y_score_matrix[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], y_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 參考 http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    y_score_matrix_ravel = y_score_matrix.ravel()
    i_positive = y_score_matrix_ravel != 0
    fpr["highest_probability"], tpr[
        "highest_probability"], micro_thresholds = roc_curve(
        y_binary.ravel()[i_positive], y_score_matrix_ravel[i_positive])
    roc_auc["highest_probability"] = auc(fpr["highest_probability"],
                                         tpr["highest_probability"])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], micro_thresholds = roc_curve(
        y_binary.ravel(), y_score_matrix.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    n_classes = len(possible_labels)
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # key_series = 'micro'
    key_series = 'highest_probability'
    i_optimal_micro = np.argmax(tpr[key_series] - fpr[key_series])
    optimal_threshold_fpr = fpr[key_series][i_optimal_micro]
    optimal_threshold_tpr = tpr[key_series][i_optimal_micro]
    optimal_threshold = micro_thresholds[i_optimal_micro]
    print('optimal_threshold_fpr:', optimal_threshold_fpr)
    print('optimal_threshold_tpr:', optimal_threshold_tpr)
    print('optimal_threshold:', optimal_threshold)

    # Plot all ROC curves
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    plt.plot(fpr["highest_probability"], tpr["highest_probability"],
             label='ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["highest_probability"]),
             color='blue', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def _run_analysis():
    checkpoint_dir_path = FLAGS.checkpoint_path
    info_file_path = os.path.join(checkpoint_dir_path, 'info.json')
    with open(info_file_path, 'r') as f:
        info = json.load(f)

    logits_list = info['logits']
    labels = info['labels']
    predictions = info['predictions']
    probabilities = info['probabilities']

    _plot_roc(logits_list, labels, predictions, probabilities)
    return


def inspect_datasets():
    examples = []
    for i in range(5):
        tfrecords_filename = os.path.join(
            DATASET_DIR,
            'plants_validation_{:05d}-of-00005.tfrecord'.format(i))
        examples.extend(inspect_tfrecords(tfrecords_filename))
    print(len(examples))
    examples = []
    for i in range(5):
        tfrecords_filename = os.path.join(
            DATASET_DIR,
            'plants_train_{:05d}-of-00005.tfrecord'.format(i))
        examples.extend(inspect_tfrecords(tfrecords_filename))
    print(len(examples))


def resize(im, target_smallest_size):
    resize_ratio = 1.0 * target_smallest_size / min(list(im.size))
    target_size = tuple(int(resize_ratio * l) for l in im.size)
    return im.resize(target_size, PIL.Image.BILINEAR)


def central_crop(im, w, h):
    half_w = im.size[0] / 2
    half_h = im.size[1] / 2
    return im.crop(
        (half_w - w / 2, half_h - h / 2, half_w + w / 2, half_h + h / 2))


def pre_process_resnet(im, coreml=False):
    target_smallest_size = 224
    im1 = resize(im, target_smallest_size)
    im2 = central_crop(im1, target_smallest_size, target_smallest_size)
    arr = np.asarray(im2).astype(np.float32)

    if not coreml:
        arr[:, :, 0] -= _R_MEAN
        arr[:, :, 1] -= _G_MEAN
        arr[:, :, 2] -= _B_MEAN
    return arr


def central_crop_by_fraction(im, central_fraction):
    w = im.size[0]
    h = im.size[1]
    return central_crop(im, w * central_fraction, h * central_fraction)


def pre_process_mobilenet(im, coreml=False):
    # 參考 https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
    # 裡的 preprocess_for_eval
    im1 = central_crop_by_fraction(im, 0.875)
    target_smallest_size = 224
    im2 = im1.resize((target_smallest_size, target_smallest_size),
                     PIL.Image.BILINEAR)
    arr = np.asarray(im2).astype(np.float32)
    if not coreml:
        arr /= 255.0
        arr -= 0.5
        arr *= 2.0
    return arr


def pre_process(im, coreml=False):
    return {
        'resnet_v2_50': pre_process_resnet,
        'mobilenet_v1': pre_process_mobilenet,
    }[model_name](im, coreml=coreml)


def _inference_by_pb():
    # http://www.cnblogs.com/arkenstone/p/7551270.html
    filenames = [
        ('20180330/1lZsRrQzj/1lZsRrQzj_5.jpg', u'通泉草'),
        ('20180330/iUTbDxEoT/iUTbDxEoT_0.jpg', u'杜鵑花仙子'),
        # ('20180330/4PdXwYcGt/4PdXwYcGt_5.jpg', u'酢漿草'),
    ]
    for filename, label in filenames:
        filename = dataset_dir_file(filename)
        # image_np = cv2.imread(filename)
        result = run_inference_on_file(filename)
        index = result['prediction_label']
        print("Prediction label index:", index)
        prediction_name = result['prediction_name']
        print("Prediction name:", prediction_name)
        print("Top 3 Prediction label index:", ' '.join(result['top_n_names']))
        assert prediction_name == label


def dataset_dir_file(filename):
    filename = os.path.join(DATASET_DIR, filename)
    return filename


def run_inference_by_pb(image_np):
    pb_file_path = '%s/frozen_graph.pb' % MODEL_DIR
    image_size = 224

    image_np = pre_process(image_np)
    image_np = cv2.resize(image_np, (image_size, image_size))
    # expand dims to shape [None, 299, 299, 3]
    image_np = np.expand_dims(image_np, 0)

    with tf.gfile.GFile(pb_file_path) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=graph) as sess:
            input_tensor_name = "input:0"
            # output_tensor_name = "resnet_v2_50/predictions/Reshape_1:0"
            output_tensor_name = OUTPUT_MODEL_NODE_NAMES_DICT[
                                     model_name] + ":0"
            input_tensor = sess.graph.get_tensor_by_name(
                input_tensor_name)  # get input tensor
            output_tensor = sess.graph.get_tensor_by_name(
                output_tensor_name)  # get output tensor
            logits = sess.run(output_tensor,
                              feed_dict={input_tensor: image_np})
            return logits


def _inference_by_coreml():
    labels_to_names = read_label_file(FLAGS.dataset_dir)
    filenames = [
        ('20180330/1lZsRrQzj/1lZsRrQzj_5.jpg', u'通泉草'),
        ('20180330/iUTbDxEoT/iUTbDxEoT_0.jpg', u'杜鵑花仙子'),
        # ('20180330/4PdXwYcGt/4PdXwYcGt_5.jpg', u'酢漿草'),
    ]
    for filename, label in filenames:
        filename = os.path.join(DATASET_DIR, filename)
        image_np = PIL.Image.open(filename)
        logits = run_inference_by_coreml(image_np)

        print('logits', logits)
        index = np.argmax(logits)
        print("Prediction label index:", index)
        prediction_name = labels_to_names[index]
        print("Prediction name:", prediction_name)
        index_list = np.argsort(logits)
        print("Top 3 Prediction label index:",
              index_list,
              ' '.join([labels_to_names[i] for i in list(index_list)]))
        assert prediction_name == label


def run_inference_by_coreml(image_np):
    frozen_model_file = '%s/frozen_graph.pb' % MODEL_DIR
    coreml_model_file = '%s/plant.mlmodel' % MODEL_DIR
    image_np = pre_process(image_np, coreml=True)

    image = Image.fromarray(image_np.astype('int8'), 'RGB')
    input_tensor_shapes = {
        "input:0": [1, image_np.shape[0], image_np.shape[1],
                    3]}  # batch size is 1
    output_tensor_name = OUTPUT_MODEL_NODE_NAMES_DICT[model_name] + ":0"

    coreml_model = coremltools.models.MLModel(coreml_model_file)

    convert_model = False
    # convert_model = True
    if convert_model:
        extra_args = {
            'resnet_v2_50': {
                'red_bias': -_R_MEAN,
                'green_bias': -_G_MEAN,
                'blue_bias': -_B_MEAN,
            },
            'mobilenet_v1': {
                'red_bias': -1.0,
                'green_bias': -1.0,
                'blue_bias': -1.0,
                'image_scale': 2.0 / 255.,
            }
        }[model_name]
        coreml_model = tfcoreml.convert(
            tf_model_path=frozen_model_file,
            mlmodel_path=coreml_model_file.replace('.mlmodel',
                                                   '_test.mlmodel'),
            input_name_shape_dict=input_tensor_shapes,
            output_feature_names=[output_tensor_name],
            image_input_names=['input:0'],
            **extra_args
        )

    coreml_inputs = {'input__0': image}
    coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=False)

    # example output: 'resnet_v2_50__predictions__Reshape_1__0'
    probs = coreml_output[
        output_tensor_name.replace('/', '__').replace(':', '__')].flatten()
    return probs


def run_inference_on_file_pb(filename):
    labels_to_names = read_label_file(FLAGS.dataset_dir)
    image_np = PIL.Image.open(filename)
    logits = run_inference_by_pb(image_np)
    index = np.argmax(logits, 1)
    prediction_name = labels_to_names[index[0]]
    index_list = np.argsort(logits, 1)
    top_n_names = list(reversed(
        [labels_to_names[i] for i in list(index_list[0])]))
    print('logits', logits)
    result = {
        'prediction_name': prediction_name,
        'prediction_label': index[0],
        'top_n_names': top_n_names,
        'logits': logits.tolist(),
    }
    return result


def run_inference_on_file(filename):
    return run_inference_on_file_pb(filename)


def main(_):
    # _inference_by_pb()
    # _inference_by_coreml()
    _run_info()


if __name__ == '__main__':
    tf.app.run()
