import gin
import logging
import wandb
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like

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

    # setup wandb
    # wandb.init(project='idrid-example', name=run_paths['model_id'],
    #            config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(group=True)
    logging.info("Dataset IDRID is successfully loaded")

    # # 下面的代码仅用于debug时查看ds中的数据是否正常
    # import matplotlib.pyplot as plt
    # # 创建一个迭代器
    # iterator = iter(ds_train)
    # # 获取一个批次（32个样本）
    # sample_batch = next(iterator)
    # # 提取图像和标签
    # sample_images = sample_batch[0]
    # sample_labels = sample_batch[1]
    # # 显示这个批次的图像
    # fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(sample_images[i].numpy())  # 假设图像是彩色的
    #     ax.set_title(f"Label: {sample_labels[i].numpy()}")
    #     ax.axis('off')
    # plt.show()

    # model
    logging.info(f"start model initialization")
    model = vgg_like(input_shape=ds_info['shape'], n_classes=ds_info['num_classes'])
    logging.info("model initialization finished")

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
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
