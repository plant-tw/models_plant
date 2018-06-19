import subprocess
from os.path import isfile, join
import time

import os
from select import select

import threading

import sys

import tensorflow as tf


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


def get_last_file(directory):
    last_file = list(sorted([f for f in os.listdir(directory)]))[-1]
    return join(directory, last_file)


start = time.time()


class RunCommandThread(threading.Thread):
    def __init__(self, target):
        super(RunCommandThread, self).__init__(target=target)
        self.daemon = True
        self._should_terminate = False

    def run_command(self, command_args):
        print('run_command', command_args)
        process = subprocess.Popen(command_args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   bufsize=1)  # line buffered

        time_limit = 1
        while True:
            poll_result = select([process.stdout], [], [], time_limit)[0]
            # print(poll_result)
            if poll_result:
                line = process.stdout.readline()
                print('{}| {}'.format(self.name, line.rstrip()))
            else:
                # print('(no output)')
                pass

            if process.poll() is not None:
                # program exited
                break

            if self._check_should_terminate():
                print('timeout, kill')
                process.kill()
                break

        rc = process.poll()
        print('rc', rc)
        print(time.time() - start)
        return rc

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
    import re
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

    def eval(self):
        call_args = self.command_args
        call_args = [a for a in call_args if
                     not a.startswith('--checkpoint_path=')]
        # ret = subprocess.call(call_args, shell=True)
        file_path = tf.train.latest_checkpoint(self.checkpoint_path)
        step = get_step(self.checkpoint_path)
        eval_dir = '{}/{}_{}'.format(self.get_eval_events_dir(),
                                     int(time.time()), step)
        mkdir_p(eval_dir)
        call_args.append('--checkpoint_path=' + file_path)
        call_args.append('--eval_dir={}'.format(eval_dir))

        self.run_command(call_args)

    def read_summary(self):
        last_event_dir = get_last_file(self.get_eval_events_dir())
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


def main():
    # run_command(['sleep', '10'])
    pretrained_checkpoint_path = 'resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
    checkpoint_path = 'resnet_v2_50_plants_0617'
    dataset_dir = '/projects/private/plant/data_non_exif'

    train_script_params = {
        'train_dir': checkpoint_path,
        'dataset_name': 'plants',
        'dataset_split_name': 'train',
        'dataset_dir': dataset_dir,
        'model_name': 'resnet_v2_50',
        'clone_on_cpu': True,
        'checkpoint_path': pretrained_checkpoint_path,
        'checkpoint_exclude_scopes': 'resnet_v2_50/logits',
        'save_summaries_secs': '120',
        'save_interval_secs': '120',
        'num_preprocessing_threads': '4',
        'trainable_scopes': 'resnet_v2_50/logits',
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
        'dataset_split_name': 'validation',
        'model_name': 'resnet_v2_50',
    }

    eval_script_args = [
                           sys.executable,
                           'research/slim/eval_image_classifier.py',
                       ] + dict_to_command_args(eval_script_params)
    train_thread = TrainThread(train_script_args)
    # train_thread.start()


    # No need to start evaluation so early
    # time.sleep(60)
    # eval_script_args = ['which', 'python']
    eval_thread = EvalThread(eval_script_args, checkpoint_path)
    # eval_thread.start()

    print('started')
    eval_every_n_step = 50
    while True:
        step = get_step(checkpoint_path)
        _train_params = train_script_params.copy()
        _train_params.update(max_number_of_steps=step + eval_every_n_step)
        train_thread.run_command(
            train_script_args + dict_to_command_args(_train_params))

        eval_thread.eval()

        # raise
        # eval_thread.join()
        # train_thread.terminate()
        # train_thread.join()


main()
