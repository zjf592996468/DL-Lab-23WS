# Team14
- Jifeng Zhou (st181366)
- Yang Cao (st186635)


# How to run the code
## P1: Diabetic Retinopathy Detection
Train default **cnn** model with IDRID dataset for **binary classification**:

`python3 main.py --train=True --multi_class=False --model='cnn'`

Train for 5-class classification:

`python3 main.py --train=True --multi_class=True --model='cnn'`

Train other models:
* vgg:

    `python3 main.py --train=True --multi_class=False --model='vgg'`

Tune hyperparameters:

`python3 wandb_sweep_cnn.py --train=True --multi_class=False --model='cnn'`

Deep visualization:

`python3 visual.py`
You can get Grad-CAM and Guided-Grad-CAM to help us to learn the model.

Transfer learning:

If run transfer.py remember **shift the train step**.

`python3 transfer.py`
You can use the efficientnet to transfer learn.

## P2: Human Activity Recognition
Train with HAPT dataset:

`python3 main.py --train=True wandb='hapt'`


# Results
We found that there is a quarter checkpoint around step 2w,
and its test accuracy can reach 90%, but we cannot find it accurately every time,
so we chose step 4w to stabilize it at 86%.

By using **transfer learning**, you can easily achieve an accuracy of about 86%,
in about 10,000 steps, which saves a lot of computing power.


# Datasets
### IDRID Dataset
The IDRID dataset is located at _/home/data/idrid_dataset_ on the GPU-Server.

### Kaggle Challenge dataset provided by EyePACS
The Kaggle Challenge dataset is part of Tensorflow Datasets.
The data loading pipeline is already part of the provided skeleton.
The tensorflow_datasets folder is located at _/home/data/tensorflow_datasets_.

### HAPT dataset
The HAPT dataset is located at _/home/data/HAPT_dataset_ on the GPU-Server.

'RawData/labels.txt': include all the activity labels available for the dataset (1 per row). 
* Column 1: experiment number ID, 
* Column 2: user number ID, 
* Column 3: activity number ID 
* Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
* Column 5: Label end point (in number of signal log samples)

### Real World (HAR) dataset
The Real World (HAR) dataset is located at _/home/data/realworld2016_dataset_
on the GPU-Server.


# P1: Diabetic Retinopathy Detection
## T1: Input_pipeline
Implement an efficient data input pipeline for the IDRID dataset. Make sure to
create a train, validation, and test split for potential hyperparameter tuning.

* ### [Optional] Profiling
If you want to analyze the performance of your input pipeline or even your
overall training performance, TensorFlow offers a Profiler.
If you’re interested in profiling your pipeline, you can start looking at the
guide https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras and
ask a tutor for further help.

## T2: Model architecture
Think of a reasonable architecture and implement it using the `tf.keras.Model`
class(https://www.tensorflow.org/api_docs/python/tf/keras/Model) in combination
with `tf.keras.layers` (https://www.tensorflow.org/api_docs/python/tf/keras/layers).
Print the model summary and check the number of parameters within each layer.

## T3: Metrics
Use common metrics that are widely used in the field of Diabetic Retinopathy
classification. Moreover, you may additionally need some metrics that reflect
the nature of your dataset, e.g. if your dataset is highly imbalanced.
For implementing any metric, use the `tf.keras.metrics.Metric` class
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric?hl=de.

## T4: Training and evaluation
* ### T4.1: Training routine
Implement a training routine that regularly logs your progress on the training
and validation data.

* ### T4.2: TensorBoard
Use TensorBoard to visualize the training progress. TensorBoard is a great way
to quickly analyze runs and compare them to others.

* ### T4.3: Saving and loading checkpoints
Save your model regularly during training so that you do not lose too much
progress in case of any failure. Moreover, implement a functionality which
allows you to continue training from a saved checkpoint/model.

* ### T4.4: Evaluation
Implement an evaluation method that loads a specified checkpoint and evaluates
your model on the test set.

## T5: Data augmentation
Extend your data input pipeline with several data augmentation operations.
Apply data augmentation online which means that you create new data on the fly
during training. Analyze to which extent a data augmentation operation has an
effect on the performance.

## T6: Deep visualization
* Visualizing activations and weights
* Dimensionality Reduction
* Gradient-based methods

Implement at least one of the following deep visualization methods.
Implement them on your own, do not use any given libraries. Use the deep
visualization method to gain deeper insights into your trained model(s)
and potentially detect failure modes.

## T7: [Optional] Transfer learning

## T8: [Optional] Ensemble learning

## T9: [Optional] Multi-class classification
