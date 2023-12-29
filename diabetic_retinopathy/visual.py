import gin
from deep_visualization.cam import grad_cam
from deep_visualization.guidcam import guided_grad_cam
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.cnnmodel import create_cnn_nets
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import app

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())
    # load dataset
    ds_train, ds_val, ds_test, ds_info = load()
    # model vgg
    # model = vgg_like(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])
    # model cnn
    model = create_cnn_nets(ds_info)
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

    for images, _ in ds_test.take(1):
        image = images[0]
        image = tf.expand_dims(image, axis=0)  # 扩展维度以符合模型输入
        break

    category_index = 0  # 这里以类别索引 0 为例
    layer_name = 'conv2d_2'  # 替换为你选择的卷积层名称

    heatmap_cam = grad_cam(model, image, category_index, layer_name)
    heatmap_guid = guided_grad_cam(model, image, category_index, layer_name)
    original_image = image[0].numpy()


    # 创建一个2行2列的子图结构
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 子图1：原图
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')  # 关闭坐标轴

    # 子图2：Grad-CAM 热力图
    axs[0, 1].imshow(original_image, alpha=0.75)  # 首先显示原图
    axs[0, 1].imshow(heatmap_cam, cmap='jet', alpha=0.5)  # 叠加显示热力图，半透明
    axs[0, 1].set_title("Grad-CAM")
    axs[0, 1].axis('off')  # 关闭坐标轴

    # 子图3：Guided Grad-CAM 热力图
    axs[1, 0].imshow(original_image, alpha=0.75)  # 首先显示原图
    axs[1, 0].imshow(heatmap_guid, cmap='jet', alpha=0.5)  # 叠加显示热力图，半透明
    axs[1, 0].set_title("Guided Grad-CAM")
    axs[1, 0].axis('off')  # 关闭坐标轴

    # 第四个子图我们不使用，可以隐藏它
    axs[1, 1].axis('off')

    # 调整子图间的空间
    plt.tight_layout()
    # 显示图形
    plt.show()

if __name__ == "__main__":
    app.run(main)
