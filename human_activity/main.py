import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from models.rnn import create_rnn
from utils import utils_params, utils_misc
import tensorflow as tf
from evaluation.visualization import visual

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('wandb', 'hapt', 'The name of the wandb project.')
flags.DEFINE_string('layer', 'Bidirectional LSTM', 'Choose which layer to use in RNN.')


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
    logging.info("Wandb logged in.")

    # setup pipeline
    ds_train, ds_val, ds_test, ds_show, ds_info = load()
    logging.info("Dataset HAPT is successfully loaded.")

    # model rnn with bi_LSTM
    logging.info("Start model initialization...")
    model = create_rnn(ds_info)
    logging.info("model-rnn initialized.")
    model.summary()

    # checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=5)

    # Load latest checkpoint
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
        evaluate(model,
                 ckpt_restore_path,
                 ds_test,
                 ds_info,
                 )

        # # Plot predicted result
        # visual(model, ckpt_restore_path, ds_show, ds_info)
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
