import subprocess
from os.path import isfile, join
import time

import os
from select import select

import threading

import sys


def read_eval_summary(path_to_events_file):
    import tensorflow as tf
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
    last_file = [f for f in os.listdir(directory)
                 if isfile(join(directory, f))][-1]
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


class EvalThread(RunCommandThread):
    def __init__(self, command_args):
        target = self.run_loop
        super(EvalThread, self).__init__(target)
        self.name = 'E'
        self.command_args = command_args

    def eval(self):
        call_args = self.command_args
        # ret = subprocess.call(call_args, shell=True)
        self.run_command(call_args)

    def read_summary(self):
        last_event_file = get_last_file('/tmp/tfmodel')
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


def main():
    # run_command(['sleep', '10'])
    pretrained_checkpoint_path = 'resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
    checkpoint_path = 'resnet_v2_50_plants_0617'
    dataset_dir = '/projects/private/plant/data_non_exif'

    train_script_args = [
        sys.executable,
        'research/slim/train_image_classifier.py',
        '--train_dir=' + checkpoint_path,
        '--dataset_name=plants',
        '--dataset_split_name=train',
        '--dataset_dir=' + dataset_dir,
        '--model_name=resnet_v2_50',
        '--clone_on_cpu',
        '--checkpoint_path=' + pretrained_checkpoint_path,
        '--checkpoint_exclude_scopes=resnet_v2_50/logits',
        '--save_summaries_secs=120',
        '--save_interval_secs=120',
        '--num_preprocessing_threads=4',
        '--trainable_scopes=resnet_v2_50/logits',
    ]

    eval_script_args = [
        sys.executable,
        'research/slim/eval_image_classifier.py',
        '--alsologtostderr',
        '--checkpoint_path=' + checkpoint_path,
        '--dataset_dir=' + dataset_dir,
        '--dataset_name=plants',
        '--dataset_split_name=validation',
        '--model_name=resnet_v2_50',
    ]

    train_thread = TrainThread(train_script_args)
    train_thread.start()
    # raise

    # No need to start evaluation so early
    time.sleep(60)
    # eval_script_args = ['which', 'python']
    eval_thread = EvalThread(command_args=eval_script_args)
    eval_thread.start()
    print('started')
    eval_thread.join()
    train_thread.terminate()
    train_thread.join()


main()
