import subprocess
from os.path import isfile, join
import time

import os
import re
from select import select

import threading

import sys

import click
import json
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import tensorflow as tf
import tfcoreml
import yaml
import matplotlib.pyplot as plt

TRAINING_SET_NAME = 'train'
VALIDATION_SET_NAME = 'validation'

OUTPUT_MODEL_NODE_NAMES_DICT = {
    'resnet_v2_50': 'resnet_v2_50/predictions/Reshape_1'
}


def read_eval_summary(path_to_events_file):
    last_summary = {}
    print(path_to_events_file)
    for e in reversed(list(tf.train.summary_iterator(path_to_events_file))):
        print('step', e.step)
        tag_simple_value_dict = {
            v.tag: v.simple_value
            for v in e.summary.value
        }
        accuracy = tag_simple_value_dict.get('eval/Accuracy')
        recall_5 = tag_simple_value_dict.get('eval/Recall_5')
        if accuracy is not None:
            print('accuracy', accuracy)
            print('recall_5', recall_5)

            return {
                'accuracy': accuracy,
                'recall_5': recall_5,
            }

        print(tag_simple_value_dict)
        # for v in e.summary.value:
        #     print(v.tag, v.simple_value)
        #     if 'loss' in v.tag:
        #         print(v.tag, v.simple_value)
        # if v.tag == 'loss' or v.tag == 'accuracy':
        #     print(v.simple_value)

        # break


def get_last_file(directory, name_filter=None):
    last_file = list(sorted([
        f for f in filter(name_filter, os.listdir(directory))
    ]))[-1]
    return join(directory, last_file)


start = time.time()


def run_command_generator(command_args, check_should_terminate=None):
    print('run_command: {}'.format(' '.join(command_args)))
    process = subprocess.Popen(command_args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               bufsize=1)  # line buffered
    time_limit = 1
    while True:
        poll_result = select([process.stdout], [], [], time_limit)[0]
        # print(poll_result)
        if poll_result:
            line = process.stdout.readline().rstrip()
            yield line, process

            if check_should_terminate and check_should_terminate(line):
                process.kill()
                break
        else:
            # print('(no output)')
            pass

        if process.poll() is not None:
            # program exited
            break

    rc = process.poll()
    print('rc', rc)
    # return rc


def run_command(command_args,
                command_params_dict=None,
                convert_line=None, check_should_terminate=None):
    convert_line = convert_line or (lambda l: l)
    if command_params_dict:
        command_args = command_args + dict_to_command_args(command_params_dict)
    for line, _ in run_command_generator(
            command_args, check_should_terminate=check_should_terminate):
        print(convert_line(line))


class RunCommandThread(threading.Thread):
    def __init__(self, target):
        super(RunCommandThread, self).__init__(target=target)
        self.daemon = True
        self._should_terminate = False

    def run_command(self, command_args, command_params_dict=None):
        print(time.time() - start)
        run_command(
            command_args,
            command_params_dict=command_params_dict,
            convert_line=lambda l: '{}| {}'.format(self.name, l),
            check_should_terminate=lambda l: self._check_should_terminate()
        )

    def _check_should_terminate(self):
        # return time.time() - start > 3
        return self._should_terminate

    def terminate(self):
        self._should_terminate = True


class TrainThread(RunCommandThread):
    def __init__(self, command_args):
        target = self.train
        super(TrainThread, self).__init__(target)
        self.name = 'T'
        self.command_args = command_args

    def train(self):
        # self.run_command(['top'])
        # self.run_command(['watch', '-n1', 'date'])
        self.run_command(self.command_args)
        pass


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_step(checkpoint_path):
    file_path = tf.train.latest_checkpoint(checkpoint_path)
    if not file_path:
        return 0

    return int(re.search('-(\d+)$', file_path).group(1))


class EvalThread(RunCommandThread):
    def __init__(self, command_args, checkpoint_path):
        target = self.run_loop
        super(EvalThread, self).__init__(target)
        self.name = 'E'
        self.command_args = command_args
        self.checkpoint_path = checkpoint_path

    def get_eval_events_dir(self):
        return '{}/eval_events'.format(self.checkpoint_path)

    def eval(self, script_params, split_name=VALIDATION_SET_NAME):
        script_params = script_params.copy()

        # ret = subprocess.call(call_args, shell=True)
        file_path = tf.train.latest_checkpoint(self.checkpoint_path)
        step = get_step(self.checkpoint_path)
        eval_dir = '{}/{}_{}_{}'.format(self.get_eval_events_dir(),
                                        int(time.time()), step, split_name, )
        mkdir_p(eval_dir)

        script_params.update(
            checkpoint_path=file_path,
            eval_dir=eval_dir,
            dataset_split_name=split_name,
        )

        self.run_command(self.command_args, script_params)

    def read_summary(self, split_name=VALIDATION_SET_NAME):
        last_event_dir = get_last_file(
            self.get_eval_events_dir(),
            name_filter=lambda x: x.endswith('_' + split_name))
        last_event_file = get_last_file(last_event_dir)
        return read_eval_summary(last_event_file)

    def run_loop(self):
        best_record = {}
        while True:
            print('run_loop loop')
            self.eval()
            # print(ret)
            summary = self.read_summary()
            print(summary)
            accuracy = summary['accuracy']
            now = time.time()
            print('now', now)
            if accuracy > best_record.get('accuracy', 0):
                best_record = {
                    'accuracy': accuracy,
                    'time': now,
                    'checkpoint': None,
                }
                print('best', best_record)

            if accuracy > 97 and now - best_record.get('time', now) > 60 * 60:
                return best_record

            print('eval sleep')
            time.sleep(3 * 60)


def dict_to_command_args(d):
    return [
        '--{}={}'.format(k, v) if v is not True else '--{}'.format(k)
        for k, v in d.items()
    ]


def run_train_eval_loop(config):
    pretrained_checkpoint_path = config['pretrained_checkpoint_path']
    checkpoint_path = config['checkpoint_path']
    dataset_dir = config['dataset_dir']
    model_name = config['model_name']
    eval_every_n_step = int(config.get('eval_every_n_step', 50))

    trainable_scopes = {
        'resnet_v2_50': 'resnet_v2_50/logits',
        'mobilenet_v1': 'MobilenetV1/Logits',
    }[model_name]
    train_script_params = {
        'train_dir': checkpoint_path,
        'dataset_name': 'plants',
        'dataset_split_name': TRAINING_SET_NAME,
        'dataset_dir': dataset_dir,
        'model_name': model_name,
        'clone_on_cpu': True,
        'checkpoint_path': pretrained_checkpoint_path,
        'checkpoint_exclude_scopes': trainable_scopes,
        'save_summaries_secs': '120',
        'save_interval_secs': '120',
        'num_preprocessing_threads': '4',
        'trainable_scopes': trainable_scopes,
    }
    train_script_args = [
        sys.executable,
        'research/slim/train_image_classifier.py',
    ]
    eval_script_params = {
        'alsologtostderr': True,
        'checkpoint_path': checkpoint_path,
        'dataset_dir': dataset_dir,
        'dataset_name': 'plants',
        'dataset_split_name': VALIDATION_SET_NAME,
        'model_name': model_name,
    }
    eval_script_args = [
        sys.executable,
        'research/slim/eval_image_classifier.py',
    ]
    train_thread = TrainThread(train_script_args)
    # train_thread.start()
    # No need to start evaluation so early
    # time.sleep(60)
    # eval_script_args = ['which', 'python']
    eval_thread = EvalThread(eval_script_args, checkpoint_path)
    # eval_thread.start()
    print('started')
    while True:
        step = get_step(checkpoint_path)
        _train_params = train_script_params.copy()
        _train_params.update(max_number_of_steps=step + eval_every_n_step)
        _train_params.update(config.get('extra_train_params') or {})
        train_thread.run_command(train_script_args, _train_params)

        eval_thread.eval(script_params=eval_script_params)
        eval_thread.eval(script_params=eval_script_params,
                         split_name=TRAINING_SET_NAME)
        summary = eval_thread.read_summary(
            split_name=VALIDATION_SET_NAME) or {}
        summary['training'] = eval_thread.read_summary(
            split_name=TRAINING_SET_NAME)

        step = get_step(checkpoint_path)
        summary['step'] = step
        summary['time'] = time.time()
        # print(summary)
        # break
        with open(get_accuracy_log_path(config), 'a+') as f:
            f.write('{}\n'.format(summary))

        do_plot(config)
        # raise
        # eval_thread.join()
        # train_thread.terminate()
        # train_thread.join()


def get_accuracy_log_path(config):
    checkpoint_path = config['checkpoint_path']
    return join(checkpoint_path, 'accuracy.log')


def export_graph(config):
    checkpoint_dir = config['checkpoint_path']
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    dataset_dir = config['dataset_dir']
    model_name = config['model_name']
    freeze_graph_script_path = config['freeze_graph_path']

    inference_graph_path = os.path.join(checkpoint_dir, 'inference_graph.pb')
    frozen_graph_path = os.path.join(checkpoint_dir, 'frozen_graph.pb')

    script_params = {
        'alsologtostderr': True,
        'model_name': model_name,
        'dataset_name': 'plants',
        'dataset_dir': dataset_dir,
        'output_file': inference_graph_path,
    }
    run_command([
        sys.executable, 'research/slim/export_inference_graph.py'
    ], command_params_dict=script_params)

    run_command([
        sys.executable, freeze_graph_script_path
    ], command_params_dict={
        'input_graph': inference_graph_path,
        'output_graph': frozen_graph_path,
        'input_checkpoint': checkpoint_path,
        'output_node_names': OUTPUT_MODEL_NODE_NAMES_DICT[model_name],
        'input_binary': 'true',
    })

    return frozen_graph_path


def export_coreml(config, frozen_graph_path):
    checkpoint_dir = config['checkpoint_path']
    model_name = config['model_name']

    output_mlmodel_path = os.path.join(checkpoint_dir, 'plant.mlmodel')
    model_extra_kwargs_dict = {
        'resnet_v2_50': {
            'red_bias': -123.68,
            'green_bias': -116.78,
            'blue_bias': -103.94,
        }

    }
    tfcoreml.convert(
        tf_model_path=frozen_graph_path,
        mlmodel_path=output_mlmodel_path,
        output_feature_names=[
            '{}:0'.format(OUTPUT_MODEL_NODE_NAMES_DICT[model_name])
        ],
        image_input_names=['input:0'],
        input_name_shape_dict={'input:0': [1, 224, 224, 3]},
        **model_extra_kwargs_dict.get(model_name, {})
    )


def unique(list_, get_key):
    result = []
    seen = {}
    for item in list_:
        key = get_key(item)
        if key in seen:
            continue

        seen[key] = True
        result.append(item)

    return result


def do_plot(config, save=True, show=False):
    checkpoint_path = config['checkpoint_path']
    with open(get_accuracy_log_path(config), 'r') as f:
        records = unique([eval(line.strip()) for line in f.readlines()],
                         lambda x: x.get('step'))

    steps = list(map(lambda x: x.get('step'), records))
    test_accuracy_list = list(map(lambda x: x.get('accuracy'), records))
    train_accuracy_list = list(
        map(lambda x: x.get('training', {}).get('accuracy'), records))

    test_accuracy_list_top5 = list(map(lambda x: x.get('recall_5'), records))
    train_accuracy_list_top5 = list(
        map(lambda x: x.get('training', {}).get('recall_5'), records))

    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.plot(steps, train_accuracy_list,
             color='r', linewidth=1.0, marker='o', label='Training')
    plt.plot(steps, test_accuracy_list,
             color='b', linewidth=1.0, marker='o', label='Validation')

    plt.plot(steps, train_accuracy_list_top5,
             color='r', dashes=[3, 1], label='Training Top-5')
    plt.plot(steps, test_accuracy_list_top5,
             color='b', dashes=[3, 1], label='Validation Top-5')

    plt.legend(loc='best')

    if save:
        img_path = join(checkpoint_path, 'accuracy.png')
        plt.savefig(img_path)
        print('Chart saved as {}'.format(img_path))

    if show:
        plt.show()


@click.command()
@click.argument('config_file')
@click.option('--export-models', is_flag=True)
@click.option('--show-plot', is_flag=True)
@click.option('--export-plot', is_flag=True)
def main(config_file, export_models, show_plot, export_plot):
    with open(config_file) as f:
        config = yaml.load(f)

    print('config: {}'.format(config))

    if show_plot:
        do_plot(config, show=True)
    elif export_plot:
        do_plot(config)
    elif not export_models:
        run_train_eval_loop(config)
    else:
        frozen_graph_path = export_graph(config)
        export_coreml(config, frozen_graph_path)


if __name__ == '__main__':
    main()
