import os

import tfcoreml

checkpoint_dir = 'resnet_v2_50_plants_0426'
frozen_graph_path = os.path.join(checkpoint_dir, 'frozen_graph.pb')
output_file_path = os.path.join(checkpoint_dir, 'plant.mlmodel')

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
tfcoreml.convert(
    tf_model_path=frozen_graph_path,
    mlmodel_path=output_file_path,
    output_feature_names=[
        'resnet_v2_50/predictions/Reshape_1:0'
    ],
    red_bias=-_R_MEAN,
    green_bias=-_G_MEAN,
    blue_bias=-_B_MEAN,
    image_input_names=['input:0'],
    input_name_shape_dict={'input:0': [1, 224, 224, 3]},
)
