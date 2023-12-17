import logging
import gin
from deep_visualization.cam import grad_cam
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.cnnmodel import create_cnn_nets
import tensorflow as tf
import matplotlib.pyplot as plt

# generate folder structures
run_paths = utils_params.gen_run_folder()


# load dataset
ds_train, ds_val, ds_test, ds_info = datasets.load(group=True)
# model vgg
# model = vgg_like(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])
# model cnn
model = create_cnn_nets(input_shape=(256,256,3), num_blocks=3, filters=8, kernel_size=(3,3), dense_units=128, dropout_rate=0.4)
# load the model
ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=5)
# 加载最新的检查点
ckpt_restore_path = manager.latest_checkpoint
print(ckpt_restore_path)
if ckpt_restore_path:
    ckpt.restore(ckpt_restore_path).expect_partial()
    print("Checkpoint restored from:", ckpt_restore_path)
else:
    print("No checkpoint found at:", run_paths['path_ckpts_train'])

for images, _ in ds_train.take(1):
    image = images[0]
    image = tf.expand_dims(image, axis=0)  # 扩展维度以符合模型输入
    break


category_index = 0  # 这里以类别索引 0 为例
layer_name = 'conv2d_2'  # 替换为你选择的卷积层名称

heatmap = grad_cam(model, image, category_index, layer_name)  # todo: Name 'image' can be undefined
plt.imshow(heatmap)
