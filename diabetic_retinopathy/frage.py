import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from evaluation.eval import evaluate1
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.kerasmodel import create_and_compile_cnn_model
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load('idrid', r'C:\Users\西门水羊\Desktop\DL Lab\idrid\IDRID_dataset')

    # model
    model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    # model = create_and_compile_cnn_model()


    checkpoint = tf.train.Checkpoint(model)

# Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time
# checkpoint.save is called, the save counter is increased.
    save_path = checkpoint.save(run_paths['path_ckpts_train'])

# Restore the checkpointed values to the `model` object.
    checkpoint.restore(save_path)


if __name__ == "__main__":
    app.run(main)
