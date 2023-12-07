import gin
import tensorflow as tf
import logging

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval):
        # Summary Writer
        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
        self.manager = tf.train.CheckpointManager(self.ckpt, run_paths['path_ckpts_train'], max_to_keep=3)

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval



    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        tf.print("在 train_step 内的损失:", loss)
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def load_checkpoint(self):
        checkpoint_dir = self.run_paths["path_ckpts_train"]
        print(f"Checking for checkpoint in: {checkpoint_dir}")
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        print(f"Latest checkpoint found: {latest_ckpt}")

        if latest_ckpt:
            self.ckpt.restore(latest_ckpt)
            print(f"Restored from {latest_ckpt}")
        else:
            print("Initializing from scratch.")

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))
                
                # Write summary to tensorboard

            if step % self.ckpt_interval == 0:
                checkpoint_path = self.run_paths["path_ckpts_train"]
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                save_path = self.manager.save()
                print("Checkpoint path:", checkpoint_path)

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                # ...
                checkpoint_path = self.run_paths["path_ckpts_train"]
                self.model.save_weights(checkpoint_path)
                logging.info(f'Saved final checkpoint to {checkpoint_path}')
                # 计算并返回验证集上的准确率
                val_accuracy = self.val_accuracy.result().numpy()
                logging.info(f'Validation accuracy: {val_accuracy * 100:.2f}%')
                return self.val_accuracy.result().numpy()

def get_checkpoint_path():
    # 返回检查点的保存路径，可以是固定的或根据某些逻辑生成的 后期可以删除
    return "path_to_checkpoints"