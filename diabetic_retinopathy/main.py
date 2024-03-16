import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from input_pipeline.preprocessing import plot_augment
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.cnnmodel import create_cnn_nets
from transfer_learning.efficientnet import transfermodel
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('multi_class', False, 'Specify whether to take multi_classification')
flags.DEFINE_boolean('classification', True, 'Specify whether to take multi_classification with classification')
flags.DEFINE_string('model', 'cnn', 'The name of the model')
flags.DEFINE_string('wandb', 'idrid', 'The name of the wandb project')
flags.DEFINE_boolean('l2_loss', True, 'Specify whether to use l2 regularization')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup wandb
    wandb.login(key="5aa34be742c563b8db6d03e8722deccc9bc0a91a")
    wandb.init(project=FLAGS.wandb, name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    logging.info("Wandb logged in")

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()
    logging.info("Dataset IDRID is successfully loaded.")

    # # plot augmentation
    # for image, label in ds_test.take(1):
    #     plot_augment(image[8])
    # logging.info("Augmentation plot is saved in results.")

    # choose model
    if FLAGS.model == 'vgg':  # model vgg
        model = vgg_like(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])
        logging.info("model-vgg_like initialized.")

    elif FLAGS.model == 'cnn':  # model cnn
        model = create_cnn_nets(ds_info=ds_info)
        logging.info("model-cnn initialized.")

    elif FLAGS.model == 'effnet':  # model efficientnet
        model = transfermodel(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])
        model.build((None, 224, 224, 3))
        logging.info("model-efficientnet initialized.")

    else:
        raise ValueError("Invalid model name. Please choose model name from 'vgg', 'cnn', 'effnet'.")
    model.summary()

    # checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=10)

    # load latest checkpoint
    ckpt_restore_path = manager.latest_checkpoint
    logging.info(f"Current checkpoint restore path: {ckpt_restore_path}")

    if FLAGS.train:
        if ckpt_restore_path:
            ckpt.restore(ckpt_restore_path).expect_partial()
            logging.info(f"Checkpoint restored from: {ckpt_restore_path}")
        else:
            logging.info(f"No checkpoint found at: {run_paths['path_ckpts_train']}")

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
        evaluate(model, ckpt_restore_path, ds_test, ds_info)
    else:
        evaluate(model, ckpt_restore_path, ds_test, ds_info)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
