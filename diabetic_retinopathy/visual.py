import gin
from deep_visualization.cam import grad_cam, overlay_heatmap
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.cnnmodel import create_cnn_nets
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import flags, app
from train import Trainer


FLAGS = flags.FLAGS
flags.DEFINE_boolean('multi_class', False, 'Specify whether to take multi_classification')
flags.DEFINE_string('model', 'cnn', 'The name of the model')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # load dataset
    ds_train, ds_val, ds_test, ds_info = load()

    # choose model
    if FLAGS.model == 'vgg':  # model vgg
        model = vgg_like(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])

    elif FLAGS.model == 'cnn':  # model cnn
        model = create_cnn_nets(ds_info=ds_info)
    model.summary()

    # load the checkpoint
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

    # Find 30 images from the test dataset that match a specified category index
    layer_name = 'max_pooling2d_2'  # Replace the max pooling layer
    category_index_1 = 1  # The label we're interested in
    images_found = 0  # Counter for images processed

    for images, labels in ds_test:
        for i, label in enumerate(labels):
            if label.numpy() == category_index_1 and images_found < 30:
                # Prepare the image
                image = images[i]  # Get the i-th image from the batch
                image_expanded = tf.expand_dims(image, axis=0)  # Expand dims to fit model input

                # Generate Grad-CAM heatmap
                heatmap = grad_cam(model, image_expanded, category_index_1, layer_name)

                # Overlay heatmap
                original_image = image.numpy()
                overlay_image = overlay_heatmap(original_image, heatmap, 0.35, 0.5)

                # Plotting
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 subplot

                # Original Image
                axs[0].imshow(original_image)
                axs[0].set_title(f'Original Image {images_found + 1}')
                axs[0].axis('off')

                # Grad-CAM Heatmap
                axs[1].imshow(heatmap)
                axs[1].set_title(f'Grad-CAM {images_found + 1}')
                axs[1].axis('off')

                # Overlay Image
                axs[2].imshow(overlay_image)
                axs[2].set_title(f'Overlay Grad-CAM {images_found + 1}')
                axs[2].axis('off')

                plt.tight_layout()
                plt.show()

                images_found += 1  # Increment the counter

                if images_found >= 30:  # Break if we've found 10 images
                    break
        if images_found >= 30:  # Check again to ensure we stop searching
            break


if __name__ == "__main__":
    app.run(main)
