# 智慧型植物辨識圖鑑app - 機器學習模型

本專案目前暫時根基於[Tensorflow Models](https://github.com/tensorflow/models/)，所以含有大量來自該專案的程式碼，日後假如能有辦法乾淨的分離的話，會考慮讓程式碼完全分離。

## 將圖檔轉為tfrecord

執行
```
research/slim/download_and_convert_plants.py --dataset_dir=[dataset路徑]
```
會在dataset路徑中產生training/evaluation用的tfrecord，以及labels.txt


## 訓練模型

用法：
`models/run_pipeline.py [yaml設定檔路徑]`

run_pipeline會讀取yaml格式的設定檔，以決定使用的模型種類、檔案存放的路徑...等。

以下是yaml設定檔範例
```
model_name: 'mobilenet_v1'
pretrained_checkpoint_path: 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
checkpoint_path: 'checkpoint_save_path'
dataset_dir: 'dataset'
freeze_graph_path: 'freeze_graph.py'
```

* model_name: 模型種類，目前支援mobilenet_v1、resnet_v2_50
* pretrained_checkpoint_path: pre-train好的模型路徑，可以從[這裡](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)取得
* checkpoint_path: 訓練過程中的checkpoint要存在哪裡
* dataset_dir: 前述tfrecord存放的路徑
* freeze_graph_path: [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)在系統中的路徑

## 輸出模型

執行
```
models/run_pipeline.py [yaml設定檔路徑] --export-models
```

會在checkpoint_path指定的路徑中產生android用的frozen_graph.
pb、plant.tflite以及iOS app用的plant.mlmodel

