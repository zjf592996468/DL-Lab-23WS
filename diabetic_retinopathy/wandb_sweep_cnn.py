import logging
import wandb
import gin
import math
import sys
from absl import flags
from input_pipeline.datasets import load
from train import Trainer
from utils import utils_params, utils_misc
from models.cnnmodel import create_cnn_nets
from models.architectures import vgg_like

FLAGS = flags.FLAGS
flags.DEFINE_boolean('multi_class', False, 'Specify whether to take multi_classification')


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder(','.join(bindings))

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load()

        # model
        model = create_cnn_nets(ds_info=ds_info)

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    'name': 'idrid-sweep',
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [50000, 55000, 40000, 45000]
        },
        'create_cnn_nets.filters': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(8),
            'max': math.log(128)
        },
        'create_cnn_nets.num_blocks': {
            'distribution': 'q_uniform',
            'q': 1,
            'min': 2,
            'max': 5
        },
        'create_cnn_nets.dense_units': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(16),
            'max': math.log(256)
        },
        'create_cnn_nets.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.8
        },
        'l2_lambda': {
            'distribution': 'uniform',
            'min': 0.001,
            'max': 0.01
        }
    }
}

if __name__ == '__main__':

    FLAGS(sys.argv)
    wandb.login(key="f27c584f9e444901abf85615134f27d2da6e411d")
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train_func, count=40)
    wandb.finish()
