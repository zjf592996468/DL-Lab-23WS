import logging
import gin
from deep_visualization.cam import grad_cam
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.cnnmodel import create_cnn_nets
import tensorflow as tf
import matplotlib.pyplot as plt


run_paths = utils_params.gen_run_folder()

# set loggers
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

# gin-config
gin.parse_config_files_and_bindings(['configs/config.gin'], [])
utils_params.save_config(run_paths['path_gin'], gin.config_str())
# load dataset
ds_train, ds_val, ds_test, ds_info = datasets.load(group=True)
# model vgg
#model = vgg_like(input_shape=[256,256,3], n_classes=2)
# model cnn
model = create_cnn_nets()
# load the model
ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
manager = tf.train.CheckpointManager(ckpt, run_paths['path_ckpts_train'], max_to_keep=5)
# 加载最新的检查点
ckpt_restore_path = manager.latest_checkpoint
print(ckpt_restore_path)
if ckpt_restore_path:
    ckpt.restore(ckpt_restore_path).expect_partial()
    print("Checkpoint restored from:", ckpt_restore_path)
else:
    print("No checkpoint found at:", run_paths['path_ckpts_train'])

for images, _ in ds_train.take(1):
    image = images[0]
    image = tf.expand_dims(image, axis=0)  # 扩展维度以符合模型输入
    break

'''Model: "vgg_like"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 256, 256, 43)      1204      
                                                                 
 conv2d_1 (Conv2D)           (None, 256, 256, 43)      16684     
                                                                 
 max_pooling2d (MaxPooling2  (None, 128, 128, 43)      0         
 D)                                                              
                                                                 
 conv2d_2 (Conv2D)           (None, 128, 128, 172)     66736     
                                                                 
 conv2d_3 (Conv2D)           (None, 128, 128, 172)     266428    
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 64, 64, 172)       0         
 g2D)                                                            
                                                                 
 global_average_pooling2d (  (None, 172)               0         
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 32)                5536      
                                                                 
 dropout (Dropout)           (None, 32)                0         
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 356654 (1.36 MB)
Trainable params: 356654 (1.36 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
'''

'''Model: "cnn_like"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 256, 256, 8)       224       
                                                                 
 batch_normalization (Batch  (None, 256, 256, 8)       32        
 Normalization)                                                  
                                                                 
 max_pooling2d (MaxPooling2  (None, 128, 128, 8)       0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 16)      1168      
                                                                 
 batch_normalization_1 (Bat  (None, 128, 128, 16)      64        
 chNormalization)                                                
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 64, 64, 16)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 64, 64, 32)        4640      
                                                                 
 batch_normalization_2 (Bat  (None, 64, 64, 32)        128       
 chNormalization)                                                
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 32, 32, 32)        0         
 g2D)                                                            
                                                                 
 global_average_pooling2d (  (None, 32)                0         
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 128)               4224      
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 2)                 258       
                                                                 
=================================================================
Total params: 10738 (41.95 KB)
Trainable params: 10626 (41.51 KB)
Non-trainable params: 112 (448.00 Byte)
_________________________________________________________________
'''
category_index = 0  # 这里以类别索引 0 为例
layer_name = 'conv2d_2'  # 替换为你选择的卷积层名称

heatmap = grad_cam(model, image, category_index, layer_name)
plt.imshow(heatmap)