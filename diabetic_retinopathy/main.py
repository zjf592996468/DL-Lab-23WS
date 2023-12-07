import gin
import logging
from absl import app, flags
from train import get_checkpoint_path
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.kerasmodel import create_and_compile_cnn_model


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load('idrid',r'/home/data/idrid_dataset')

    #creat checkpoints path
    checkpoint_path = get_checkpoint_path()

    # 创建 run_paths 字典
    run_paths = {"path_ckpts_train": checkpoint_path}
    #model
    #model = vgg_like(input_shape=(256, 256, 3), n_classes=2)
    model= create_and_compile_cnn_model()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        trainer.load_checkpoint()
        for _ in trainer.train():
            continue
    else:
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)

if __name__ == "__main__":
    app.run(main)
