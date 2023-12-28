import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from evaluation.eval import evaluate1
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.cnnmodel import create_cnn_nets
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

    # setup wandb
    wandb.login(key="f27c584f9e444901abf85615134f27d2da6e411d")
    wandb.init(project='idrid-cnn-cy', name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    logging.info("Wandb logged in")

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()
    logging.info("Dataset IDRID is successfully loaded")

    # # 下面的代码仅用于debug时查看ds中的数据是否正常
    # import matplotlib.pyplot as plt
    # import numpy as np
    # # 创建一个迭代器
    # iterator = iter(ds_train)
    # # 设置子图的行和列数
    # rows, cols = 4, 8
    # # 显示所有批次的图像
    # for sample_batch in iterator:
    #     # 提取图像和标签
    #     sample_images = sample_batch[0]
    #     sample_labels = sample_batch[1]
    #     # 获取当前批次的图像数量和批次大小
    #     batch_size = len(sample_images)
    #     # 计算调整后的行和列数
    #     adjusted_rows = int(np.ceil(batch_size / cols))
    #     adjusted_cols = min(batch_size, cols)
    #     # 显示这个批次的图像
    #     fig, axes = plt.subplots(adjusted_rows, adjusted_cols, figsize=(16, 8))
    #     for i, ax in enumerate(axes.flat):
    #         if i < batch_size:
    #             ax.imshow(sample_images[i].numpy())  # 假设图像是彩色的
    #             ax.set_title(f"Label: {sample_labels[i].numpy()}")
    #             ax.axis('off')
    #         else:
    #             ax.axis('off')  # 隐藏多余的子图
    #     plt.show()

    # model vgg
    logging.info("start model initialization")
    # model = vgg_like(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])
    # logging.info("model-vgg_like initialized")

    # model cnn
    model = create_cnn_nets(ds_info)
    logging.info("model-cnn initialized")
    logging.info(model.summary())

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
        evaluate1(model, ds_test, run_paths)
    else:
        evaluate(model,
                 ckpt_restore_path,
                 ds_test,
                 run_paths
                 )
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
