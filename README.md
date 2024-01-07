# Team14
- Jifeng Zhou (st181366)
- Yang Cao (st186635)


# How to run the code
## P1: Diabetic Retinopathy Detection
- Train and evaluate model with IDRID dataset on **binary classification**

  Type `python3 main.py --train=True --multi_class=False` in batch file.
- 5-class classification

  Type `python3 main.py --train=True --multi_class=True` in batch file.
  
  if run transfer.py remenber shift the train step

# Results
to do


# Datasets
### IDRID Dataset
The IDRID dataset is located at _/home/data/idrid_dataset_ on the GPU-Server.

### Kaggle Challenge dataset provided by EyePACS
The Kaggle Challenge dataset is part of Tensorflow Datasets.
The data loading pipeline is already part of the provided skeleton.
The tensorflow_datasets folder is located at _/home/data/tensorflow_datasets_.

### HAPT dataset
The HAPT dataset is located at _/home/data/HAPT_dataset_ on the GPU-Server.

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
