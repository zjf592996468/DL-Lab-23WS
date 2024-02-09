# Team14

- Jifeng Zhou (st181366)
- Yang Cao (st186635)

# How to run the code
**Attention!** The checkpoint logic of our code is different from the original code. 
We are, if there is a checkpoint it is loaded. 

So please rename the `ckpt` file before running it if you don't want to use checkpoint.
Equally if you want to use a good checkpoint just drag the file into `enperiments` and rename it to ckpt (p2: har-ckpt)
## P1: Diabetic Retinopathy Detection

Train default '**cnn**' model with **L2-regularization** on IDRID dataset for binary classification:

run batch file: '**run.sh**'

Train other model:

* vgg:
  `python3 main.py --train=True --multi_class=False --model='vgg'`

Tune hyperparameters:

`python3 wandb_sweep_cnn.py --train=True --multi_class=False --model='cnn'`

Deep visualization:

`python3 visual.py`

Transfer learning:

  `python3 wandb_sweep_cnn.py --train=True --multi_class=False --model='effnet --l2_loss=False'`

Train '**cnn**' model for multi-class classification:

* Use classification:
  `python3 main.py --train=True --multi_class=True --model='cnn'`

* Use regression:
  `python3 main.py --train=True --multi_class=True --model='cnn' --classification=False`

## P2: Human Activity Recognition

Train RNN model with '**Bidirectional LSTM**' layer on HAPT dataset:

`python3 main.py --train=True`

Train RNN model with other RNN layer:

* LSTM:
  `python3 main.py -- train=True --layer='LSTM'`

* Bidirectional GRU:
  `python3 main.py --train=True --layer='Bidirectional GRU'`

* GRU:
  `python3 main.py --train=True --layer='GRU'`

# Results

## Binary Model Test Accuracy
| Model                  | CNN | VGG | EfficientNet* |
|------------------------|-----|-----|---------------|
| **Test Accuracy (%)**  | 89  | 74  | 88            |

## Multi-Classification Model Test Accuracy

| Model                  | CNN-Classification | CNN-Regression |
|------------------------|--------------------|----------------|
| **Test Accuracy (%)**  | 57                 | 43             |

*The best record of EfficientNet is achieved by
using the last 80% as the TRAIN set and the first 20% as the VALIDATION set.


## Human Activity Recognition
| Model                     | RNN with bidirectional LSTM |
|---------------------------|-----------------------------|
| **Test Accuracy(%)**      | 95                          |
| **Balanced Accuracy(%)**  | 83                          |


We found that there is a quarter checkpoint around step 2w,
and its test accuracy can reach 89%, but we cannot find it accurately every time,
so we chose step 4w to stabilize it at 86%.

By using **transfer learning**, you can easily achieve an accuracy of about 86%,
in about 10,000 steps, which saves a lot of computing power.
