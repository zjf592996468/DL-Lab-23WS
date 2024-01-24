import gin
from deep_visualization.cam import grad_cam, overlay_heatmap
from deep_visualization.guidcam import guided_grad_cam
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import vgg_like
import train
from models.cnnmodel import create_cnn_nets
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import flags, app
from train import Trainer
import matplotlib

FLAGS = flags.FLAGS
flags.DEFINE_boolean('multi_class', False, 'Specify whether to take multi_classification')


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
    # load latest checkpoint
    ckpt_restore_path = manager.latest_checkpoint
    print(ckpt_restore_path)
    if ckpt_restore_path:
        ckpt.restore(ckpt_restore_path).expect_partial()
        print("Checkpoint restored from:", ckpt_restore_path)
    else:
        print("No checkpoint found at:", run_paths['path_ckpts_train'])

    category_index_0 = 0  # Specified category index
    layer_name = 'max_pooling2d_2'  # Replace with the name of the convolutional layer of your choice

    # Find images from the test dataset that match a specified category index
    for images, labels in ds_test:
        for i, label in enumerate(labels):
            if label.numpy() == category_index_0:
                image_0 = images[i]
                image_0 = tf.expand_dims(image_0, axis=0)  # Extending dimensions to match model inputs
                found = True
                break
        if found:
            break
    category_index_1 = 1  # Specified category index

    # Find images from the test dataset that match a specified category index
    for images, labels in ds_test:
        for i, label in enumerate(labels):
            if label.numpy() == category_index_1:
                image_1 = images[i]
                image_1 = tf.expand_dims(image_1, axis=0)  # Extending dimensions to match model inputs
                found = True
                break
        if found:
            break


    heatmap_cam_0 = grad_cam(model, image_0, category_index_0, layer_name)
    heatmap_cam_1 = guided_grad_cam(model, image_1, category_index_1, layer_name)
    original_image_0 = image_0[0].numpy()
    original_image_1 = image_1[0].numpy()

    overlay_image = overlay_heatmap(orig_image=original_image_0, heatmap=heatmap_cam_0)
    overlay_image1 = overlay_heatmap(orig_image=original_image_1, heatmap=heatmap_cam_1)
    # Create a 2x3 subplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Displays the original image and the superimposed heat map on the first line
    axs[0, 0].imshow(original_image_0)
    axs[0, 0].set_title('Original Image_0')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(heatmap_cam_0)
    axs[0, 1].set_title('Grad-CAM_0')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(overlay_image)
    axs[0, 2].set_title('Overlayed Grad-CAM_0')
    axs[0, 2].axis('off')

    # Displays the original image, the Guided Grad-CAM, and the superimposed Guided Grad-CAM in the second row.
    axs[1, 0].imshow(original_image_1)
    axs[1, 0].set_title('Original Image_1')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(heatmap_cam_1)
    axs[1, 1].set_title('Grad-CAM_1')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(overlay_image1)
    axs[1, 2].set_title('Overlayed Grad-CAM_1')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
    matplotlib.use('Agg')


if __name__ == "__main__":
    app.run(main)
