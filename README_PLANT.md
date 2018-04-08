# Plant Model

用Tensorflow Slim 提供的pre-train過的model來訓練模型

## Download Pre-trained Model

Download the model "ResNet V2 50" - http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
from https://github.com/tensorflow/models/tree/master/research/slim

## Training

research/slim/train_image_classifier.py
--train_dir=resnet_v2_50_plants_non_exif
--dataset_name=plants
--dataset_split_name=train
--dataset_dir={DATESET_DIR}
--model_name=resnet_v2_50
--clone_on_cpu
--checkpoint_path=resnet_v2_50_2017_04_14/resnet_v2_50.ckpt
--checkpoint_exclude_scopes=resnet_v2_50/logits
--save_summaries_secs=120
--save_interval_secs=120
--num_preprocessing_threads=4
--trainable_scopes=resnet_v2_50/logits

## Evaluation
research/slim/eval_image_classifier.py
--alsologtostderr
--checkpoint_path=resnet_v2_50_plants_non_exif/
--dataset_dir={DATESET_DIR}
--dataset_name=plants
--dataset_split_name=validation
--model_name=resnet_v2_50

## Jupyter notebook:

plant.ipynb