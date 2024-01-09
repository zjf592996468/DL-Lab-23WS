import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('wandb', 'hapt-debug', 'The name of the wandb project')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup wandb
    wandb.login(key="f27c584f9e444901abf85615134f27d2da6e411d")
    wandb.init(project=FLAGS.wandb, name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    logging.info("Wandb logged in")

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()
    logging.info("Dataset HAPT is successfully loaded")

    # todo: implement rnn model here
    # # model rnn
    # logging.info("start model initialization")
    # model = rnn()
    # logging.info("model-rnn initialized")
    # logging.info(f"{model.summary()}")

    # checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=5)
    # 加载最新的检查点
    ckpt_restore_path = manager.latest_checkpoint
    print(ckpt_restore_path)

    if FLAGS.train:
        if ckpt_restore_path:
            ckpt.restore(ckpt_restore_path).expect_partial()
            print("Checkpoint restored from:", ckpt_restore_path)
        else:
            print("No checkpoint found at:", run_paths['path_ckpts_train'])
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
        # evaluate1(model, ds_test, run_paths)
    else:
        evaluate(model,
                 ckpt_restore_path,
                 ds_test,
                 run_paths
                 )
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
