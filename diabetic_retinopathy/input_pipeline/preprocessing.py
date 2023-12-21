import gin
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def check_imb(labels):
    """check and plot imbalance situation, return num of classes and num of samples in each class"""
    # Calculate the num of each class
    label_counts = labels["Retinopathy grade"].value_counts().sort_index()

    # Set different color for each class
    # colors = plt.cm.viridis(np.linspace(0, 1, label_counts.index.shape[0]))
    colors = plt.cm.tab10(range(label_counts.index.shape[0]))  # Options: Accent, tab10, Paired

    # Plot the figure
    plt.figure(figsize=(8, 6))
    bars = plt.bar(label_counts.index, label_counts.values, color=colors, label='Class percentages')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(label_counts.index)

    # Show num on the bar
    for i, count in enumerate(label_counts.values):
        plt.text(label_counts.index[i], count, str(count), ha='center', va='bottom')

    # Show the legend with percent info
    total_samples = sum(label_counts.values)
    percents = [count / total_samples * 100 for count in label_counts.values]
    legend_labels = [f'Class {label_counts.index[i]}: {percents[i]:.2f}%' for i in range(label_counts.index.shape[0])]
    plt.legend(bars, legend_labels, bbox_to_anchor=(1, 1))
    plt.tight_layout()

    return plt


@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""
    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32) / 255.

    # Resize image with pad to avoid distortions
    image = tf.image.resize_with_pad(image, img_height, img_width)

    return image, label


@gin.configurable()
def augment(image, label, img_height, img_width):
    """Data augmentation"""
    operations = [
        'Rotation90', 'Rotation180', 'Rotation270', 'Flippinglr', 'Flippingud', 'Cropping', 'Shearing',
        'AdjustContrast', 'AdjustBrightness']
    # Randomly use operations and at least use half of them
    # chosen_op = random.sample(operations, random.randint(round(2 / 3 * len(operations)), len(operations)))
    chosen_op = random.sample(operations, len(operations))

    for operation in chosen_op:
        if operation == 'Rotation90':
            image = tf.image.rot90(image)
        elif operation == 'Rotation180':
            image = tf.image.rot90(image, 2)
        elif operation == 'Rotation270':
            image = tf.image.rot90(image, 3)
        elif operation == 'Flippinglr':
            image = tf.image.flip_left_right(image)
        elif operation == 'Flippingud':
            image = tf.image.flip_up_down(image)
        elif operation == 'Cropping':
            # 随机裁剪并调整大小至256x256
            cropped_size = [tf.random.uniform([], minval=180, maxval=256, dtype=tf.int32) for _ in range(2)]
            image = tf.image.random_crop(image, size=[cropped_size[0], cropped_size[1], 3])
            image = tf.image.resize_with_pad(image, img_height, img_width)
        # elif operation == 'Shearing':
        #     # 利用仿射变换进行剪切，保持图像大小不变
        #     shear_x = random.uniform(-0.3, 0.3)  # x轴剪切幅度
        #     shear_y = random.uniform(-0.3, 0.3)  # y轴剪切幅度
        #     image = tfa.image.transform(image, [1.0, shear_x, 0.0, shear_y, 1.0, 0.0, 0.0, 0.0],
        #                                 interpolation='NEAREST')
        # elif operation == 'AdjustContrast':
        #     # 随机调整对比度
        #     contrast_factor = random.uniform(0.5, 1.5)  # 可根据需要调整这个范围
        #     image = tf.image.adjust_contrast(image, contrast_factor)
        # elif operation == 'AdjustBrightness':
        #     # 随机调整亮度
        #     brightness_delta = random.uniform(-0.3, 0.3)  # 可根据需要调整这个范围
        #     image = tf.image.adjust_brightness(image, brightness_delta)

    return image, label
