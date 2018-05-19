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
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_utils
from datasets.plants import read_label_file
from nets import resnet_v2
from tensorflow.python.training import monitored_session

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

CHECKPONT_PATH = '/tmp/tfmodel/'
# CHECKPONT_PATH = 'resnet_v2_50_plants_non_exif'
CHECKPONT_PATH = 'resnet_v2_50_plants_0426'
MODEL_DIR = os.environ.get('MODEL_DIR') or CHECKPONT_PATH
tf.app.flags.DEFINE_string(
    'checkpoint_path', CHECKPONT_PATH,
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

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50', 'The name of the architecture to evaluate.')

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


def get_info(checkpoint_path=None):
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
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
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
        capacity=5 * FLAGS.batch_size)

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
        'predictions': predictions,
    }


def main(_):
    info = get_info()
    checkpoint_path = info['checkpoint_path']
    num_batches = info['num_batches']
    names_to_updates = info['names_to_updates']
    variables_to_restore = info['variables_to_restore']

    feed_dict = {}
    y, _ = info['network_fn'](info['images'], reuse=True)
    with get_monitored_session(checkpoint_path) as sess:
        params = {
            k: v
            for k, v in info.items()
            if isinstance(v, tf.Tensor)
        }
        params.update(
            y=y,
        )
        res = sess.run(params, feed_dict=feed_dict)

        print(res.keys())
        print(res['predictions'])
        print(res['labels'])
        print([x == y for x, y, in zip(res['predictions'], res['labels'])])

    return

    # slim.evaluation.evaluate_once(
    #     master=FLAGS.master,
    #     checkpoint_path=checkpoint_path,
    #     logdir=FLAGS.eval_dir,
    #     num_evals=num_batches,
    #     eval_op=list(names_to_updates.values()),
    #     variables_to_restore=variables_to_restore)


def resize(im, target_smallest_size):
    resize_ratio = 1.0 * target_smallest_size / min(list(im.size))
    target_size = tuple(int(resize_ratio * l) for l in im.size)
    return im.resize(target_size, PIL.Image.BILINEAR)


def central_crop(im, w, h):
    half_w = im.size[0] / 2
    half_h = im.size[1] / 2
    return im.crop(
        (half_w - w / 2, half_h - h / 2, half_w + w / 2, half_h + h / 2))


def pre_process(im, shift=True):
    target_smallest_size = 224
    im1 = resize(im, target_smallest_size)
    im2 = central_crop(im1, target_smallest_size, target_smallest_size)
    arr = np.asarray(im2).astype(np.float32)

    if shift:
        arr[:, :, 0] -= _R_MEAN
        arr[:, :, 1] -= _G_MEAN
        arr[:, :, 2] -= _B_MEAN
    return arr


def _inference_by_pb():
    # http://www.cnblogs.com/arkenstone/p/7551270.html
    filenames = [
        ('20180330/1lZsRrQzj/1lZsRrQzj_5.jpg', u'通泉草'),
        ('20180330/4PdXwYcGt/4PdXwYcGt_5.jpg', u'酢漿草'),
    ]
    for filename, label in filenames:
        filename = os.path.join(DATASET_DIR, filename)
        # image_np = cv2.imread(filename)
        result = run_inference_on_file(filename)
        index = result['prediction_label']
        print("Prediction label index:", index)
        prediction_name = result['prediction_name']
        print("Prediction name:", prediction_name)
        print("Top 3 Prediction label index:", ' '.join(result['top_n_names']))
        assert prediction_name == label


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
            output_tensor_name = "resnet_v2_50/predictions/Reshape_1:0"
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
        ('20180330/4PdXwYcGt/4PdXwYcGt_5.jpg', u'酢漿草'),
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
    image_np = pre_process(image_np, shift=False)
    image_size = 224

    image = Image.fromarray(image_np.astype('int8'), 'RGB')
    input_tensor_shapes = {
        "input:0": [1, image_size, image_size, 3]}  # batch size is 1
    output_tensor_names = ['resnet_v2_50/predictions/Reshape_1:0']

    coreml_model = coremltools.models.MLModel(coreml_model_file)

    convert_model = False
    if convert_model:
        coreml_model = tfcoreml.convert(
            tf_model_path=frozen_model_file,
            mlmodel_path=coreml_model_file.replace('.mlmodel',
                                                   '_test.mlmodel'),
            input_name_shape_dict=input_tensor_shapes,
            output_feature_names=output_tensor_names,
            image_input_names=['input:0'],
            red_bias=-_R_MEAN,
            green_bias=-_G_MEAN,
            blue_bias=-_B_MEAN,
        )

    coreml_inputs = {'input__0': image}
    coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=False)
    probs = coreml_output['resnet_v2_50__predictions__Reshape_1__0'].flatten()
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
    _inference_by_pb()
    _inference_by_coreml()


if __name__ == '__main__':
    tf.app.run()
