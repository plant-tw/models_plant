import os

from os.path import isdir

from os import listdir

import tensorflow as tf

# import download_and_convert_plants  # noqa
from download_and_convert_plants import FLAGS, split_dataset_by_directory, \
    save_filenames_by_split, _write_dataset_info_file, SPLIT_NAME_TRAIN, \
    SPLIT_NAME_VALIDATION, _convert_dataset, _groupby_unsorted, split_dataset, \
    normalize_class_names
from datasets import dataset_utils


def _listdir_rel_and_abs_path(d):
    return [(f, os.path.join(d, f)) for f in os.listdir(d)]


def _get_filenames_and_classes(dataset_dir):
    filenames = []
    class_names = []
    for class_name, class_path in _listdir_rel_and_abs_path(dataset_dir):
        class_name = class_name.decode('utf-8')
        if not isdir(class_path):
            continue

        for _, file_path in _listdir_rel_and_abs_path(class_path):
            if not any(file_path.lower().endswith(ext) for ext in
                       ['.png', '.jpg', '.jpeg', '.gif']):
                continue

            filenames.append((file_path, class_name))
            class_names.append(class_name)

    class_names = normalize_class_names(class_names)
    return filenames, class_names


def split_dataset_by_class(photo_filenames):
    def _get_class(tuple):
        return tuple[1]

    training_filenames = []
    validation_filenames = []

    for class_name, tuples in _groupby_unsorted(photo_filenames,
                                                key=_get_class):
        _training_pairs, _validation_pairs = split_dataset(list(tuples))

        training_filenames.extend(_training_pairs)
        validation_filenames.extend(_validation_pairs)

    return training_filenames, validation_filenames


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # if _dataset_exists(dataset_dir):
    #     print('Dataset files already exist. Exiting without re-creating them.')
    #     return

    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    # print(photo_filenames, class_names)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    training_filename_pairs, validation_filename_pairs = split_dataset_by_class(
        photo_filenames)
    save_filenames_by_split(dataset_dir, training_filename_pairs,
                            validation_filename_pairs)

    v_set = set([a[1] for a in validation_filename_pairs])
    print(len(v_set))
    # return

    # Write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    _write_dataset_info_file({
        SPLIT_NAME_TRAIN: len(training_filename_pairs),
        SPLIT_NAME_VALIDATION: len(validation_filename_pairs),
    }, dataset_dir)

    # Convert the training and validation sets.
    _convert_dataset(SPLIT_NAME_TRAIN, training_filename_pairs,
                     class_names_to_ids,
                     dataset_dir)
    _convert_dataset(SPLIT_NAME_VALIDATION, validation_filename_pairs,
                     class_names_to_ids,
                     dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Plant dataset!')


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    run(FLAGS.dataset_dir)


if __name__ == '__main__':
    tf.app.run()
